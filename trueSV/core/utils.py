




import os
import pysam 
import glob as gb  
import numpy as np
import cv2
import time
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib



            
#@njit        
def extract_indel_flags(cigar_tuple, seq_ascii):

    seq_no_ins = []
    seq_with_ins = [] 
    
    read_index = 0
    for op, length in cigar_tuple:
            
        #if op in [0, 3, 4, 6, 7, 8]: # cigar tuple: 0->M, 3->N, 4->S, 6->P, 7->=, 8->X
        if op not in [1, 2]:
            seq_temp = seq_ascii[read_index:read_index + length]
            seq_no_ins.extend(seq_temp) 
            seq_with_ins.extend(seq_temp)

            read_index += length 

        elif op == 1: # 1 in cigar tuple is equal to INS
            seq_temp = seq_ascii[read_index:read_index + length]
            seq_with_ins.extend(seq_temp)
            
            read_index += length

        elif op==2: # 2 in cigar tuple is equal to DEL
            seq_temp = [200]*length if length >= 40 else [0]*length
            seq_no_ins.extend(seq_temp) 
            seq_with_ins.extend(seq_temp)



    return seq_no_ins, seq_with_ins




def extract_indel_flags_new(cigar_tuple, seq_ascii):

    seq_flags = np.ones((100000,), dtype=np.uint8)
    seq_flags[:] = 160 # channel 0 is filled with default value of 160
    
    read_index = 0
    seq_no_ins_idx = 0
    seq_with_ins_idx = 50000

    for op, length in cigar_tuple:
            
        #if op in [0, 3, 4, 6, 7, 8]: # cigar tuple: 0->M, 3->N, 4->S, 6->P, 7->=, 8->X
        if op not in [1, 2]:
            #seq_flags[seq_no_ins_idx: seq_no_ins_idx+length] = seq_ascii[read_index:read_index + length]
            #seq_flags[seq_with_ins_idx: seq_with_ins_idx+length] = seq_ascii[read_index:read_index + length]

            read_index += length
            seq_no_ins_idx += length
            seq_with_ins_idx += length

        elif op == 1: # 1 in cigar tuple is equal to INS
            if length >= 40:
                seq_flags[seq_with_ins_idx: seq_with_ins_idx+length] = seq_ascii[read_index:read_index + length].copy()

            read_index += length
            seq_with_ins_idx += length

        elif op==2: # 2 in cigar tuple is equal to DEL
            if length >=40:    
                seq_flags[seq_no_ins_idx: seq_no_ins_idx+length] = 200
                seq_flags[seq_with_ins_idx: seq_with_ins_idx+length] = 200


            seq_no_ins_idx += length
            seq_with_ins_idx += length


    return seq_flags[:seq_no_ins_idx], seq_flags[50000:seq_with_ins_idx]






def rescale_255(arr):
    arr[arr == 2] = 40
    arr[arr == 3] = 80
    arr[arr == 4] = 120
    arr[arr == 5] = 160
    arr[arr == 9] = 200
    return arr




#@njit
def find_insertions(cigar_tuple, seq_with_ins, split_sup_strand):
    insertions =  []
    index_no_ins, index_with_ins = 0, 0
    for op, length in cigar_tuple:
        if op == 1:
            start_no_ins, end_no_ins = index_no_ins, index_no_ins+length # that's correct don't change it to index_with_ins
            if length >=40:
                seq = seq_with_ins[index_with_ins: index_with_ins+length]
                
                #ins_flag = np.array([250]*length, dtype=np.uint8)
                ins_flag = np.array(seq) # we put the sampe read seq as INS flag

                split_sup_strand_flag = np.array([split_sup_strand]*length, dtype=np.uint8)
                insertions.append((start_no_ins, length, seq, ins_flag, split_sup_strand_flag))
                index_with_ins += length       
        #elif op in [0, 2, 3, 4, 6, 7, 8]: 
        else:
            index_with_ins += length
            index_no_ins += length

    return insertions



#@njit('uint8[:](unicode_type)')
# @jit
def seq2num(seq):
    #trans_table = str.maketrans('NnAaTtCcGg', '0022334455')
    trans_table = str.maketrans('NnMRAaTtCcGg', '000022334455') # 'M' was added to handle 'M' in chromosome 3
    seq_num = np.array(list(seq.translate(trans_table)), dtype=np.uint8)
    seq_num = rescale_255(seq_num)
    # np.int8 can store number less than <=127, otherwise, it will overwrite
    return seq_num

#@njit
def combine_qs_mq(qs, mq):
    pqs = np.power(10, (-1*qs)/10, dtype=np.float32)
    pmq = np.power(10,  (-1*mq)/10, dtype=np.float32)
    qs_mq = np.asanyarray(-10*np.log10(pqs+pmq-pqs*pmq), dtype=np.uint8)    
    return qs_mq


def resize_img(img, max_coverage, width=1280, height=1280):
    img_new = np.zeros(shape=(height, width, img.shape[2]), dtype=np.uint8)
    img = cv2.resize(img, dsize=(width, max_coverage), interpolation=cv2.INTER_LANCZOS4)
    img_new[:max_coverage, :, :] = img
    
    return img_new


def extract_features(reads):
    features_all = []
    for read in reads:
        read_pos = read[0]
        read_seq = read[1]
        cigar_tuple = read[2]
        split, sup, strand = read[3:6]
        border_start, img_shape = read[6], read[7]

        split_sup_strand = ((split << 2) | (sup << 1) | strand ) * 20 + 100 # flag for split, suplementary, and reverese reads 
        
        skips = cigar_tuple[0][1] if cigar_tuple[0][0]==4 else 0 # how many skips exist in the begining of the cigar string
        skips_tail = cigar_tuple[-1][1] if cigar_tuple[-1][0]==4 else 0 # how many skips exist in the begining of the cigar string
        
       
        seq_num_no_ins, seq_with_ins = extract_indel_flags_new(cigar_tuple, read_seq)
       
        insertions = find_insertions(cigar_tuple, seq_with_ins, split_sup_strand)

        #split_sup_strand_no_ins = [split_sup_strand]*len(seq_num_no_ins)
        split_sup_strand_no_ins = np.zeros_like(seq_num_no_ins)
        split_sup_strand_no_ins[:] = split_sup_strand
        

        # start and end columns, along with start and end of the read
        read_start, read_end = 0, len(seq_num_no_ins)
        col_start = read_pos - border_start - skips
        col_end = col_start + read_end


        if col_start < 0 :
            read_start = -1 * col_start
            col_start = 0
        
        if col_end > img_shape[1]: # bug got fixed 
            read_end = read_end - (col_end-img_shape[1]) # bug got fixed
            col_end = img_shape[1]
        

        del_flags = seq_num_no_ins.copy()
        del_flags[del_flags!=200] = 0
        del_flags[:skips] = 150 # leading skipps are considered as 150 value
        if skips_tail:
            del_flags[-skips_tail:] = 100 # trailing skips are considered as 100 value
        

        features_all.append((read_start, read_end, col_start, col_end, seq_num_no_ins, del_flags, split_sup_strand_no_ins, insertions))
    

    return features_all


 
#@njit
def image_maker(reads, img_shape, ref_seq, overal_cov, top_margin=2, cov_thick=10, max_cov=150):
    
    img = np.zeros(shape=img_shape, dtype=np.uint8)
    img[1, :len(ref_seq), 0] = ref_seq

    img[overal_cov + 10: overal_cov + 10 + cov_thick, :, :] = 255
    img[max_cov+overal_cov+10: max_cov+overal_cov+10+cov_thick, :, :] = 255



    features_all = extract_features(reads)

    features_all.sort(key=lambda tup:tup[2])

   
    for features in features_all:
        #t1 = time.time()
        read_start, read_end, col_start, col_end, seq_num_no_ins, del_flags, split_sup_strand_no_ins, insertions = features
        try: 
            row_no_ins = list(img[top_margin:150, col_start, 0]).index(0) + top_margin # bug got fixed
        except:
            continue
        

        if col_end > 20:
            """
            # in some situations, if the read_pos starts before border_start and \
            # the INS sequence inside the read is too long \
            # the col_end might be still negative (<0)
            # for example: 
            #               border_start = 150000
            #               read_pos = 140000
            #               skips = 2000
            #               len(seq_with_ins) =  12000
            #               len(ins) = 4000
            #               read_end = len(seq_num_no_ins) = 8000
            #               ---------------------------
            #               col_start = 14000 - 150000 - 2000 = -12000
            #               col_end = col_start + read_end = -12000 + 8000 = -4000
            # 
            # Therefore, we have to check if col_end > 0
            """
            img[row_no_ins, col_start: col_end, 0] = seq_num_no_ins[read_start: read_end] # channel 0 of the image
            img[row_no_ins, col_start: col_end, 1] = del_flags[read_start: read_end] # channel 1 of the image
            img[row_no_ins, col_start: col_end, 2] = split_sup_strand_no_ins[read_start: read_end] # channel 2 of the image
        
        
        for ins in insertions:
            start_ins, length = ins[0] - read_start, ins[1]
            seq_ins = ins[2]
            flag_ins = ins[3]
            split_sup_strand_ins = ins[4]

            col_start_ins = col_start + start_ins
            #print(f'col_start_ins {col_start_ins}, len {length}')
            if col_start_ins >= img.shape[1]:
                break 
            if col_start_ins + length > img.shape[1]:
                length = img.shape[1] - col_start_ins
            col_end_ins = col_start_ins + length

            if col_end_ins <=0: 
                continue
            # print(f'{col_start_ins} - {col_end_ins}')

            try: 
                row_ins = list(img[150:, col_start_ins, -1]).index(0) + 150
            except: 
                row_ins = False
                for shift in range(1, 11):
                    if not img[150:, col_start_ins+shift, -1].min():
                        row_ins = list(img[150:, col_start_ins+shift, 0]).index(0) + 150
                        break 
                if not row_ins:
                    continue

            img[row_ins, col_start_ins: col_end_ins, 0] = img[1, col_start_ins: col_end_ins, 0].copy() # channel 0 is filled with ref sequences
            img[row_ins, col_start_ins: col_end_ins, 1] = flag_ins[:length] # chnnale 1 is filled with inserted sequences in read
            img[row_ins, col_start_ins: col_end_ins, 2] = split_sup_strand_ins[:length]
    
    
    cov = np.sum(img[:150, :, 0]>150, axis=0) - 10
    img[0, :, 0] = cov


    return img


#sajad 
def image_saver_old(aln_path, ref_path, vcf_path, cov, chrs, max_coverage, window_size
                , overlap, max_start_at_pos, cut_size, cut_overlap, cut_resize, out_path):
    
    os.makedirs(out_path, exist_ok=True)

    aln = pysam.AlignmentFile(aln_path)
    ref = pysam.FastaFile(ref_path)
    
    num_channels = 6 
    img = np.zeros(shape=(max_coverage, window_size+overlap, num_channels), dtype=np.uint8) 
    
    aln = pysam.AlignmentFile(aln_path)
    ref = pysam.FastaFile(ref_path)
    
    for chr in chrs:
        chr_length = len(ref[chr]) # length of the chromosome
        borders = int(chr_length//window_size) #number of borders or windows
        for b in range(0, borders+1):
            border_start = 0 if b==0 else b*window_size-overlap
            border_end = (b+1)*window_size + overlap if b==0 else (b+1)*window_size
            
            img_temp1 = img[2:, -overlap:, :].copy()
            img = np.zeros(shape=(max_coverage, border_end-border_start, num_channels), dtype=np.uint8)            
            img[2:, :overlap, :] = img_temp1

            ### !!! Bug Bug -> the length of the sequence for the latest border is shorter than the window_size, so
            ### so, it cannot run this code : img[1, :, 0] = ref_seq
            ref_seq = seq2num(str(ref.fetch(chr, start=border_start, end=border_end))) # load ref_seq and convert to digits
            
            img[1, :len(ref_seq), 0] = ref_seq # put ref_seq in the first row of img
            img[1, :len(ref_seq), -1] = 1 # write '1' in the last channel to indicate that those rows have been written 


            # coverage line with 10 pixels thickness
            img[cov+10: cov+10+10, :, :-1] = 255
            img[150+cov+10: 150+cov+10+10, :, :-1] = 255
            img[cov+10: cov+10+10, :, -1] = 1
            img[150+cov+10: 150+cov+10+10, :, -1] = 1
            
            reads_info = []  
            for read in aln.fetch(contig=chr, start=border_start, stop=border_end):
                read_pos = int(read.pos)
                if read.is_unmapped or  read.is_secondary:
                    continue
                if read_pos < border_start or read_pos > border_end - overlap :
                    continue
                if not read.query_qualities:
                    continue

                read_mq = int(read.mapping_quality)
                if read_mq <20: 
                    continue

                read_seq = str(read.query_sequence)
                read_seq = list(read_seq.upper().encode('ascii'))
                read_qs = list(read.query_qualities)
                
                read_cigar = None
                read_cig_tuple = read.cigartuples
                read_rev = read.is_reverse
                read_split = read.has_tag("SA")
                read_sup = read.is_supplementary
                reads_info.append((read_pos, read_seq, read_qs, read_mq, read_cigar, read_cig_tuple, read_rev, read_split, read_sup, border_start, img.shape))
                #poses.append(read_pos)
                #mqs.append(read_mq)

            img = image_maker(reads=reads_info, img=img, offset=border_start) # img.shape -> (max_coverage, 1e6, 6) 

            aln_name = aln_path.split('/')[-1][:-4] # bamfile name 

            pos = 0
            img_cut = img[:, pos:pos+cut_size, 1:4].copy() # tha last channel is removed. img_cut.shape -> (max_coverage, 1e4, 3)
            img_cut = resize_img(img_cut, max_coverage, width=cut_resize, height=cut_resize)

            name = aln_name + '_' + chr+ '_' + str(border_start+pos) + '-' + str(border_start+pos+cut_size)
            path = out_path + name + '.png'
            cv2.imwrite(path, img_cut)

            #np.save(path, img_cut)
            print(f'chr{chr} {border_start+pos}-{border_start+pos+cut_size} saved')
            
            pos = pos + cut_size - cut_overlap # start pos for the second cut
            while pos + cut_size <= window_size:
                img_cut = img[:, pos:pos+cut_size, 1:4] #.copy()
                img_cut = resize_img(img_cut, max_coverage, width=cut_resize, height=cut_resize)

                name = aln_name + '_' + chr+ '_' + str(border_start+pos) + '-' + str(border_start+pos+cut_size)
                path = out_path + name + '.png'
                cv2.imwrite(path, img_cut) 
                print(f'chr{chr} {border_start+pos}-{border_start+pos+cut_size} saved')
                pos = pos + cut_size - cut_overlap
                




def draw_bbox(img, img_start, img_end, chrom, vcf_path):
    
    all_SVs = joblib.load(vcf_path)
    for sv in all_SVs:
        sv_chr, sv_start, sv_len, sv_type, sv_gt = sv[0], sv[1], sv[2], sv[3], sv[4]
        if sv_chr == chrom:
            sv_end = sv_start + sv_len
            if sv_start >= img_start and sv_end<=img_end:
                x0, y0 = sv_start - img_start, 0
                x1, y1 = sv_end - img_start, img.shape[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), (225, 225, 255), 3)
                print('BBOX--BBOX', img.shape, img_start, img_end, x0, y0, x1, y1 )
            
            elif sv_start < img_start and sv_end>img_start and sv_end <=img_end:
                x0, y0 = 0, 0
                x1, y1 = sv_end - img_start, img.shape[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 3)
                print('BBOX--BBOX', img_start, img_end, x0, y0, x1, y1 )
            elif sv_start >= img_start and sv_start<img_end-500 and sv_end > img_end:
                x0, y0 = sv_start - img_start, 0 
                x1, y1 = img.shape[1], img.shape[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 3)
                print('BBOX--BBOX', img_start, img_end, x0, y0, x1, y1 )
    return img






def check_indel_flag(cigar_tuple, size_thresh=20):
    count_del, count_ins = 0, 0
    for op, length in cigar_tuple:
        if op==1 and length >=size_thresh:
            count_ins+=1
        elif op==2 and length>=size_thresh:
            count_del += 1
    return count_ins, count_del






