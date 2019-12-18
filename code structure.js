__init__.py {
	from . import models
	from . import metrics
	from . import utils
}

models.py 930{
	_sparse_linear()
	_sparse_sd(){
		sklearn.linear_model.LinearRegression()
	}
	class Sparse{
		__init__()
		run(){
			_sparse_linear()
			skimage.transform.AffineTransform()
		}
	}
	class SparseSD{
		__init__()
		run(){
			_sparse_sd()
			skimage.transform.AffineTransform()
		}		
	}
	_fill_holes()
	_calculate_of(){
		cv2.optflow.createOptFlow_Farneback()
		cv2.optflow.createOptFlow_DIS()
		cv2.optflow.createOptFlow_DeepFlow()
	}
	_advection_constant_vector()
	_advection_semi_lagrangian()
	_interpolator()
	class Dense{
		__init__()
		run()
	}
	class DenseRotation{
		__init__()
		run()		
	}
	class Persistence{
		__init__()
		run()	
	}
}

utils.py 145{
	depth2intensity()
	intensity2depth()
	RYScaler()
	inv_RYScaler()
}
metrics.py 500{
	R()
	R2()
	RMSE()
	MAE()
	prep_clf()
	CSI()
	FAR()
	POD()
	HSS()
	ETS()
	BSS()
	ACC()
	precision()
	recall()
	FSC()
	MCC()
	ROC_curve()
	PR_curve()
	AUC()
}