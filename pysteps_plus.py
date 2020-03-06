
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts.steps import _compute_incremental_mask,_check_inputs,_compute_sprog_mask
from pysteps import cascade, extrapolation, noise, utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.nowcasts import utils as nowcast_utils
import scipy.ndimage
import sys
import numpy as np

def forecast12(R, V, n_timesteps, n_ens_members=24, n_cascade_levels=6,
             R_thr=None, kmperpixel=None, timestep=None,
             extrap_method="semilagrangian", decomp_method="fft",
             bandpass_filter_method="gaussian", noise_method="nonparametric",noise_stddev_adj=None, 
             ar_order=2, vel_pert_method="bps",
             conditional=False, probmatching_method="cdf",
             mask_method="incremental", callback=None, return_output=True,
             seed=None, num_workers=1, fft_method="numpy", domain="spatial", measure_time=False):

    filter_kwargs = dict()
    noise_kwargs = dict()
    extrap_kwargs = dict()

    if vel_pert_method == "bps":
        vp_par  = noise.motion.get_default_params_bps_par()
        vp_perp = noise.motion.get_default_params_bps_perp()

    num_ensemble_workers = n_ens_members if num_workers > n_ens_members else num_workers
    fft = utils.get_method(fft_method, shape=R.shape[1:], n_threads=num_workers)
    M, N = R.shape[1:]

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method((M, N), n_cascade_levels, **filter_kwargs)
    #decomp_method = cascade.get_method(decomp_method)
    decomp_method, recomp_method = cascade.get_method(decomp_method)
    extrapolator_method = extrapolation.get_method(extrap_method)
    x_values, y_values = np.meshgrid(np.arange(R.shape[2]),np.arange(R.shape[1]))
    xy_coords = np.stack([x_values, y_values])
    R = R[-(ar_order + 1):, :, :].copy()
    MASK_thr = None

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    extrap_kwargs = dict()
    extrap_kwargs['xy_coords'] = xy_coords
    res = list()

    for i in range(ar_order):  # 2
        R[i, :, :] = extrapolator_method(R[i, :, :], V, ar_order - i,"min", **extrap_kwargs)[-1]

    # get methods for perturbations
    init_noise, generate_noise = noise.get_method(noise_method)
    # initialize the perturbation generator for the precipitation field
    pp = init_noise(R, fft_method=fft, **noise_kwargs)
    noise_std_coeffs = np.ones(n_cascade_levels)

    # compute the cascade decompositions of the input precipitation fields
    R_d = []
    for i in range(ar_order + 1):
        # R_ = decomp_method(R[i, :, :], filter, MASK=MASK_thr, fft_method=fft)
        R_ = decomp_method(R[i, :, :], filter, mask=MASK_thr, fft_method=fft,
                   output_domain=domain, normalize=True,
                   compute_stats=True, compact_output=True)
        R_d.append(R_)

    # normalize the cascades and rearrange them into a four-dimensional array
    # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    #R_c, mu, sigma = nowcast_utils.stack_cascades(R_d, n_cascade_levels)
    #R_d = None
    R_c = nowcast_utils.stack_cascades(R_d, n_cascade_levels)
    R_d = R_d[-1]
    R_d = [R_d.copy() for j in range(n_ens_members)]

    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((n_cascade_levels, ar_order))
    for i in range(n_cascade_levels):
        #R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order + 1)])
        #GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK_thr)
    #R_c_ = None
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c[i], mask=MASK_thr)

    # adjust the lag-2 correlation coefficient to ensure that the AR(p) process is stationary
    for i in range(n_cascade_levels):
        GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(GAMMA[i, 0], GAMMA[i, 1])

    # estimate the parameters of the AR(p) model from the autocorrelation coefficients
    PHI = np.empty((n_cascade_levels, ar_order + 1))
    for i in range(n_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
        
    # discard all except the p-1 last cascades because they are not needed for the AR(p) model
    #R_c = R_c[:, -ar_order:, :, :]
    R_c = [R_c[i][-ar_order:] for i in range(n_cascade_levels)]

    # stack the cascades into a five-dimensional array containing all ensemble members
    #R_c = np.stack([R_c.copy() for i in range(n_ens_members)])
    R_c = [[R_c[j].copy() for j in range(n_cascade_levels)] for i in range(n_ens_members)]

    # initialize the random generators
    randgen_prec = []
    randgen_motion = []
    np.random.seed(seed)
    for j in range(n_ens_members):
        rs = np.random.RandomState(seed)
        randgen_prec.append(rs)
        seed = rs.randint(0, high=1e9)
        rs = np.random.RandomState(seed)
        randgen_motion.append(rs)
        seed = rs.randint(0, high=1e9)

    init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)
    # initialize the perturbation generators for the motion field
    vps = []
    for j in range(n_ens_members):
        kwargs = {"randstate": randgen_motion[j], "p_par": vp_par, "p_perp": vp_perp}
        vp_ = init_vel_noise(V, 1. / kmperpixel, timestep, **kwargs)
        vps.append(vp_)

    D = [None for j in range(n_ens_members)] # very important, store locations of last time step
    res = np.zeros(shape = (n_ens_members,V.shape[1], V.shape[2]),dtype = 'float32')

    MASK_prec = R[-1, :, :] >= R_thr
    # get mask parameters
    mask_rim =  10
    mask_f =  1
    # initialize the structuring element
    struct = scipy.ndimage.generate_binary_structure(2, 1)
    # iterate it to expand it nxn
    n = mask_f * timestep / kmperpixel
    struct = scipy.ndimage.iterate_structure(struct, int((n - 1) / 2.))
    # initialize precip mask for each member
    MASK_prec = _compute_incremental_mask(MASK_prec, struct, mask_rim)
    MASK_prec = [MASK_prec.copy() for j in range(n_ens_members)]

    fft_objs = []
    for i in range(n_ens_members):
        fft_objs.append(utils.get_method(fft_method, shape=R.shape[1:]))

    R = R[-1, :, :]
    #print("Starting nowcast computation.")
    # iterate each time step
    for t in range(n_timesteps):
        sys.stdout.flush()
        # iterate each ensemble member
        for j in range(n_ens_members):
            # generate noise field
            EPS = generate_noise(pp, randstate=randgen_prec[j],fft_method=fft_objs[j],domain=domain)
            # decompose the noise field into a cascade
            EPS = decomp_method(EPS, filter, fft_method=fft_objs[j],input_domain=domain, output_domain=domain,
                                    compute_stats=True, normalize=True,compact_output=True)

            # iterate the AR(p) model for each cascade level
            for i in range(n_cascade_levels):
                # normalize the noise cascade
                #EPS_ = (EPS["cascade_levels"][i, :, :] - EPS["means"][i]) / EPS["stds"][i]
                EPS_ = EPS["cascade_levels"][i]
                EPS_ *= noise_std_coeffs[i]
                # R_c[j, i, :, :, :] = autoregression.iterate_ar_model(R_c[j, i, :, :, :], PHI[i, :], EPS=EPS_)
                R_c[j][i] = autoregression.iterate_ar_model(R_c[j][i], PHI[i, :], eps=EPS_)
            del EPS, EPS_ 

            # compute the recomposed precipitation field(s) from the cascades obtained from the AR(p) model(s)
            #R_c_ = nowcast_utils.recompose_cascade(R_c[j, :, -1, :, :], mu, sigma)
            R_d[j]["cascade_levels"] = [R_c[j][i][-1, :] for i in range(n_cascade_levels)]
            R_d[j]["cascade_levels"] = np.stack(R_d[j]["cascade_levels"])
            R_c_ = recomp_method(R_d[j])

            # apply the precipitation mask to prevent generation of new precipitation into areas where it was not originally observed
            R_cmin = R_c_.min()
            R_c_ = R_cmin + (R_c_ - R_cmin) * MASK_prec[j]
            MASK_prec_ = R_c_ > R_cmin

            # Set to min value outside of mask
            R_c_[~MASK_prec_] = R_cmin
            # adjust the CDF of the forecast to match the most recently observed precipitation field
            R_c_ = probmatching.nonparam_match_empirical_cdf(R_c_, R)
            MASK_prec[j] = _compute_incremental_mask(R_c_ >= R_thr, struct, mask_rim)

            # compute the perturbed motion field
            V_ = V + generate_vel_noise(vps[j], (t + 1) * timestep)
            # advect the recomposed precipitation field to obtain the forecast for time step t
            extrap_kwargs.update({"D_prev": D[j], "return_displacement": True})
            R_f_, D_ = extrapolator_method(R_c_, V_, 1, **extrap_kwargs)
            D[j] = D_   # very important for next time step calculation
            if t == n_timesteps -1:
                res[j] = R_f_[0]
    return res