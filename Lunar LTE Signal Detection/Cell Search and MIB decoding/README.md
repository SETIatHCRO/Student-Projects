# Cell Search and MIB docoding

MATLAB code, used for manipulation of physical properties of LTE signals. 
Used to perform a Cell Search algorithm, to extract Cell ID and decodes Master Information Block.

[CellIDMIBDetect.m](CellIDMIBDetect.m) reads in an interleaved binary file, cuts this file into datasets each containing two LTE frames worth of samples, and performs Cell Search and MIB decoding attempt on every dataset.

[Freq_Sweep.m](Freq_Sweep.m) uses a MATLAB to USRP USB interface, to tune the USRPS LO in a sweeping fashion, to perform Cell Search and MIB decoding for different frequencies in one go. Otherwise the procedure is the same as in [CellIDMIBDetect.m](CellIDMIBDetect.m).