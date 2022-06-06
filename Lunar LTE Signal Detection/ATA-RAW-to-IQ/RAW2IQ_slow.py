import numpy as np
import matplotlib.pyplot as plt
from guppi import guppi
from numba import njit
import rich
import os
import contextlib
import argparse
from scipy import signal
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import sigmf


def main():
    # Initializing Parser
    parser = argparse.ArgumentParser(description="Convert the ATA's beamformer output raw file to IQ")
    parser.add_argument('-fc', '--f_c', type=float, help='Frequency which will be shifted to DC in MHz. Default is '\
                         'the center of the recorded band (48MHz). Has to be a multiple of the channel bandwidth+',\
                            nargs='?')

    parser.add_argument('-decimation', '--decim', type=int, help='Decimation factor. Decreases outbut bandwidth to reduce file size.'\
                        ' Has to be an integer larger or equal to 2.', nargs='?', default=2)

    parser.add_argument('-X', '--X-Pol', help='Choose X polarisation to process. This is the default option.', action='store_true', default=True)
    parser.add_argument('-Y', '--Y-Pol', help='Choose Y polarisation to process.', action='store_true', default=False)

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--Input_File_Path', type=str, help='Enter Path to input raw file', required=True)
    requiredNamed.add_argument('-o', '--Output_File_Path', type=str, help='Enter Path to interleaved IQ data and the name you wish the file to have (without file'+\
                        ' format extention)', required=True)
    
    args = parser.parse_args()
    if args.f_c == None:
        args.f_c = 48
    # Reading the first header to extract some information
    fname = args.Input_File_Path
    f = guppi.Guppi(fname)
    hdr = f._parse_header()
    
    # Check if parsed arguments meet requirements
    if args.decim < 1:
        raise Exception('Decimation Factor "'+str(args.decim)+'", is invalid')

    if args.Y_Pol:
        pol = 1
        pol_str = 'Y'
    else:
        pol = 0
        pol_str = 'X'


    # Read in *.raw file
    g = guppi.Guppi(fname)
    # Initialize index
    rf_sample_idx = np.array(0)
    # Calculate number of available blocks, using file size, header size and number of bits per sample
    c = hdr['NBITS']*4*hdr['PIPERBLK']*hdr['NCHAN']
    file_size = (os.path.getsize(fname))*8
    max_blocks = round(file_size/(c+hdr['HEADER_SIZE']*8))

    # Getting desired amount of blocks to process
    num_blocks = int(input('How many Blocks do you want to process? \nEach Block consists of '+\
                            str(hdr['TBIN']*hdr['PIPERBLK']*1e3)+'ms worth of data.\nThe amount'+\
                             ' of available Blocks is '+str(max_blocks)+': '))

    # Check if the desired amount of blocks is valid
    if num_blocks > max_blocks or num_blocks<=0:
        raise Exception('Amount of Blocks: "'+str(num_blocks)+'", is invalid')
    
    # Initialize array for Nyquist Samples
    ts_rf = np.zeros(hdr['PIPERBLK']*hdr['NCHAN']*2*num_blocks, dtype='float')
    f_c = np.float64(float(args.f_c)*1e6)

    # Looping through the blocks
    for block_idx in range(num_blocks):
        print('Current Block: '+str(block_idx+1))
        with contextlib.redirect_stdout(None):
            # Reading in next block
            hdr, data = g.read_next_block()
        

        for spectrum_idx in range(data.shape[1]):
            # Looping through the Spectra in one block, converting it to IQ and then to time domain
            ts_buf = np.fft.irfft(np.append(data[:, spectrum_idx, pol],0))
            for sample in ts_buf:
                # Append time domain array
                ts_rf[rf_sample_idx] = sample
                rf_sample_idx += 1

    # Shift to IQ
    ts_IQ = rf_to_IQ(ts_rf, hdr['OBSBW']*1e6*2, f_c, hdr['PIPERBLK'], hdr['NCHAN'], num_blocks)
    

    # Decimate if applicable
    if args.decim > 1:
        ts_IQ = signal.decimate(ts_IQ, args.decim, ftype='fir')

    # Calculating information for sigmf meta data
    bandwidth = hdr['OBSBW']/args.decim*2
    num_samples = len(ts_IQ)

    # Interleave and write binaries to file
    ts_IQ = np.complex64(ts_IQ)
    ts_IQ = interleave(ts_IQ)
    ts_IQ.tofile(args.Output_File_Path+'.sigmf-data', 'bw')

    
    center_frequency = hdr['OBSFREQ'] + hdr['CHAN_BW']/2 - (hdr['OBSBW']/2-args.f_c)

    meta = SigMFFile(
    data_file=args.Output_File_Path+'.sigmf-data',
    global_info = {
        SigMFFile.DATATYPE_KEY: get_data_type_str(ts_IQ),
        SigMFFile.SAMPLE_RATE_KEY: (round(bandwidth, 2)),
        SigMFFile.AUTHOR_KEY: 'Sebastian Obernberger',
        SigMFFile.DESCRIPTION_KEY: 'Interleaved ATA Beamformer IQ capture.',
        SigMFFile.VERSION_KEY: sigmf.__version__,
        SigMFFile.FREQUENCY_KEY: center_frequency,
        'Number of Blocks': str(num_blocks),
        'Number of Samples': str(num_samples),
        'Observation Time': str(hdr['TBIN']*hdr['PIPERBLK']*num_blocks*1e3)+'ms',
        'Polarisation': pol_str,
        'Original RAW header': hdr,
    }
    
    )
    assert meta.validate()
    meta.tofile(args.Output_File_Path+'.sigmf-meta')



    print('Final Bandwidth is: '+str(round(bandwidth, 2))+'MHz. Data has been shifted by: '\
            +str(round(args.f_c, 2))+'MHz. Numbers of samples: '+str(len(ts_IQ))+'. Polarisation: '+pol_str+'.')

    
@njit
def rf_to_IQ(ts_rf, sr, f_c, piperblk, nchan, num_blocks):
    ts_IQ = np.zeros(piperblk*nchan*2*num_blocks, dtype=np.complex128)
    IQ_sample_idx = 0
    for n, rf_sample in enumerate(ts_rf):
        t = (n+1)/sr
        ts_IQ[IQ_sample_idx] = rf_sample * np.exp(-1j*2*np.pi*f_c*t)
        IQ_sample_idx += 1

    

    return ts_IQ

def interleave(ts_IQ):
    output = np.append(ts_IQ.real, ts_IQ.imag)
    output[::2] = ts_IQ.real
    output[1::2] = ts_IQ.imag
    output = np.array(output, dtype='float32')
    return output

if __name__ == "__main__":
    
    main()
