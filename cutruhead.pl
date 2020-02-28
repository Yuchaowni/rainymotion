#!/usr/bin/perl
# Skip ReUseable data header
# Copyright(C) 1999 by Yuh.N
# $Id: cutruhead,v 1.1 2000/03/09 07:33:52 yuh Exp $

open(DATA, $ARGV[0]) || die $!;

# Skip until ^Z
while(read(DATA, $buf, 1) > 0) {
  $x = unpack('c', $buf);
  if ($x == 0x1a) {
	last;
  }
}

while (read(DATA, $buf, 8192) > 0) {
  print $buf;
}

exit 0;