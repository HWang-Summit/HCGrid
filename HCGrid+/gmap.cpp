//----------------------------------------------------------------
//
// title                  :gmap.cpp
// description            :\
//                          set grid kernel & healpix;
//                          read input points;
//                          read output map;
//                          set wcs for output pixels.
//                          write output map.
//                          write reordered map.
// author                 :Qi Luo
//
//----------------------------------------------------------------

#include <hdf5.h>
#include <fitsio.h>
#include <wcslib/wcslib.h>
#include "gmap.h"
#include <string>
//#include "H5Cpp.h"

double *h_lons;
double *h_lats;
double **h_data;
double *h_weights;
uint64_t *h_hpx_idx;
uint32_t *h_inx_idx;
uint32_t *h_start_ring;
uint32_t *h_zyx;
double *h_xwcs;
double *h_ywcs;
double **h_datacube;
double **h_weightscube;

char *h_header;
double *h_kernel_params;
double last_sphere_radius = -1;
double last_hpxmaxres = -1;
//bool kernel_set = false;
GMaps h_GMaps;

/* Set grid kernel & healpix lookup table. */
void _prepare_grid_kernel(uint32_t kernel_type, double *kernel_params, double sphere_radius, double hpx_max_res){
    uint64_t num_params;    // Number of parameters

    // Set kernel type & parameters
    if(kernel_type == GAUSS1D){// params = ('0.5 / kernel_sigma ** 2',)
        num_params = 1;
        kernel_params[0] = 0.5 / pow(kernel_params[0], 2.);
        h_GMaps.bearing_needed = false;
    }
    else if(kernel_type == GAUSS2D){// params = ('kernel_sigma_maj', 'kernel_sigma_min', 'PA')
        num_params = 3;
        h_GMaps.bearing_needed = true;
    }
    else if(kernel_type == TAPERED_SINC){// params = ('kernel_sigma', 'param_a', 'param_b')
        num_params = 3;
        h_GMaps.bearing_needed = false;
    }

    if (kernel_params+num_params-1 == NULL)
        UTIL_FAIL("Number Kernel Paramters Error.");

    h_GMaps.kernel_type = kernel_type;
    h_kernel_params = kernel_params;
//    kernel_set = true;

    // Recompute healpix lookup table in case kernel sphere has changed
    if(fabs(last_sphere_radius-sphere_radius) > 3e-5 || fabs(last_hpxmaxres-hpx_max_res) > 3e-5){
        uint64_t nside = set_optimal_nside(D2R * hpx_max_res);
        _Healpix_init(nside, RING);
        last_sphere_radius = sphere_radius;
        last_hpxmaxres = hpx_max_res;
//        printf("* Recompute healpix lookup table success!\n");
    }

    h_GMaps.sphere_radius = sphere_radius;
    h_GMaps.disc_size = (D2R * sphere_radius + h_Healpix._resolution);
}

// read input coordinate
void read_input_coordinate(const char *infile){
    hid_t file_id;// hid_t是HDF5对象id通用数据类型，每个id标志一个HDF5对象
    herr_t status; // herr_t是HDF5报错和状态的通用数据类型
    hid_t coords;
    // Number of input points.
    uint32_t numOfImputPoints;
    // 打开HDF5文件
    // 文件id = H5Fopen(const char *文件�?
    //                  unsigned 读写flags,
    //                    - H5F_ACC_RDWR可读可写    //                    - H5F_ACC_RDONLY只读 
    //                  hid_t 访问性质)
    file_id = H5Fopen(infile, H5F_ACC_RDWR, H5P_DEFAULT); 

    // Number of input points.
    // uint32_t numOfImputPoints = 10000000; //竖扫ver19个文件每个文件采样点�?3660*19=69540 , 横扫hor 18*3876=69768
    // h_GMaps.data_shape = numOfImputPoints;
    // h_GMaps.spec_dim = 10;
    
    //分配内存空间

    // h_data = RALLOC(double, numOfImputPoints);
    // h_lons = RALLOC(double, numOfImputPoints); //longitude赤经ra
    // h_lats = RALLOC(double, numOfImputPoints); //latitude赤纬dec
    // h_weights = RALLOC(double, numOfImputPoints); //采样权重

    coords = H5Gopen(file_id, "coords", H5P_DEFAULT);

    // 读取属性 (num_samples)
    hid_t attr_id;
    attr_id = H5Aopen_name(coords, "num_samples");
    status = H5Aread(attr_id, H5T_NATIVE_INT, &numOfImputPoints);
    // printf("numofinput is: %d\n", numOfImputPoints);
    h_GMaps.data_shape = numOfImputPoints;
    h_GMaps.spec_dim = 5;

    //分配内存空间
    h_lons = RALLOC(double, numOfImputPoints); //longitude赤经ra
    h_lats = RALLOC(double, numOfImputPoints); //latitude赤纬dec
    h_weights = RALLOC(double, numOfImputPoints); //采样权重
    // Read the coordinate
    // 创建数据集中的数据本�?   // dataset_id = H5Dopen(group位置id,
    //                 const char *name, 数据集名
    //                    数据集访问性质)
    hid_t dataset_id;
    // Read the coordinates
    //赤经
    dataset_id = H5Dopen(coords, "xcoords", H5P_DEFAULT); 
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_lons);
    //赤纬
    dataset_id = H5Dopen(coords, "ycoords", H5P_DEFAULT); 
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_lats);

    // Intial the weight
    for(int i=0; i<numOfImputPoints; ++i)
        h_weights[i] = 1.;
    // 关闭dataset相关对象
    status = H5Dclose(dataset_id);
    // 关闭文件对象
    status = H5Fclose(file_id);
}
void read_input_data(const char *infile){
    hid_t file_id;// hid_t是HDF5对象id通用数据类型，每个id标志一个HDF5对象
    herr_t status; // herr_t是HDF5报错和状态的通用数据类型
    hid_t signal;
    // 打开HDF5文件
    file_id = H5Fopen(infile, H5F_ACC_RDWR, H5P_DEFAULT); 
    signal = H5Gopen(file_id, "signal", H5P_DEFAULT);
    hid_t dataset_id;
    string sig = "signal_";
    char buff[32];
    for(int i=0; i<h_GMaps.spec_dim;i++){
        sprintf(buff,"%d",i);
        string tmp = string(buff);
        // printf("buffer is %s\n",buff);
        sig+=tmp;
        //strcpy(buff,sig.c_str());
        dataset_id = H5Dopen(signal, sig.c_str(), H5P_DEFAULT); 
        sig = "signal_";
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_data[i]);
    }
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
}


/* 3. Read output map. fitsfile*/
void read_output_map(const char *infile){
    long naxes, *naxis;
    int status=0, nkeyrec, nfound;
    fitsfile *fptr;

    // Open FITS file
    fits_open_file(&fptr, infile, READONLY, &status);
    printfitserror(status);

    // Read FITS header
    fits_hdr2str(fptr, 0, NULL, 0, &h_header, &nkeyrec, &status);
    printfitserror(status);

    // Read the dimension
    fits_read_key_lng(fptr, "NAXIS", &naxes, NULL, &status);
    printfitserror(status);
    naxis = RALLOC(long, naxes);
    fits_read_keys_lng(fptr, "NAXIS", 1, naxes, naxis, &nfound, &status);
    printfitserror(status);
    UTIL_ASSERT(nfound==naxes, "nfound!=naxes");
    h_zyx = RALLOC(uint32_t, naxes);
    for(int i=0; i<naxes; ++i)
        h_zyx[i] = (uint32_t) naxis[naxes-i-1];
    h_GMaps.block_warp_num = (h_zyx[1] - 1) / (32 * h_GMaps.factor) + 1;

    // Release
    DEALLOC(naxis);
    fits_close_file(fptr, &status);
    printfitserror(status);
}

/* Set wcs for output pixels. */
int set_WCS(){
    int naxes = 3;
    int status=0, nkeyrec, nreject, nwcs, tempInd;
    struct wcsprm *wcs;
    int *stat;
    double *pixcrd, *imgcrd, *phi, *theta, *world;

    // Get the dimension
    uint32_t z = h_zyx[0];
    uint32_t y = h_zyx[1];
    uint32_t x = h_zyx[2];
    uint64_t ncoords = x * y;

    // Parse the primary header of the FITS file
    nkeyrec = strlen(h_header) / 80;
    if((status = wcspih(h_header, nkeyrec, WCSHDR_all, 2, &nreject, &nwcs, &wcs))){
        fprintf(stderr, "wcspih ERROR %d: %s.\n", status, wcshdr_errmsg[status]);
    }

    // Initialize the wcsprm struct
    if ((status = wcsset(wcs))) {
        fprintf(stderr, "wcsset ERROR %d: %s.\n", status, wcs_errmsg[status]);
        return 0;
    }

    // Set target map pixels.
    pixcrd = RALLOC(double, ncoords*naxes);
    tempInd = 0;
    for(int yy=0; yy<y; ++yy){
        for(int xx=0; xx<x; ++xx){
            pixcrd[tempInd] = xx+1;
            pixcrd[tempInd+1] = yy+1;
            pixcrd[tempInd+2] = z;
            tempInd += naxes;
        }
    }

    // Transfrom target pixels to sky coordinates
    imgcrd = RALLOC(double, ncoords*naxes);
    world = RALLOC(double, ncoords*naxes);
    phi = RALLOC(double, ncoords);
    theta = RALLOC(double, ncoords);
    stat = RALLOC(int, ncoords);
    h_xwcs = RALLOC(double, ncoords);
    h_ywcs = RALLOC(double, ncoords);
                         
    if ((status = wcsp2s(wcs, ncoords, naxes, pixcrd, imgcrd, phi, theta, world, stat))){
        fprintf(stderr, "\n\nwcsp2s ERROR %d: %s.\n", status, wcs_errmsg[status]);
    }

    tempInd = 0;
    for(int i=0; i<x*y; ++i) {
        h_xwcs[i] = world[tempInd];
        h_ywcs[i] = world[tempInd + 1];
        tempInd += naxes;
    }

    // Release
    DEALLOC(pixcrd);
    DEALLOC(imgcrd);
    DEALLOC(phi);
    DEALLOC(theta);
    DEALLOC(stat);
    DEALLOC(world);
    status = wcsvfree(&nwcs, &wcs);

    return 1;
}

/* Write output map. */
void write_output_map(const char *infile){
    fitsfile *fptr;
    int naxis, ncoords, status = 0, nkeyrec, i, j, k, first;
    long *naxes;
    bool end;
    char keyrec[81];

    // Create output FITS file, delete any pre-existing file
    fits_create_file(&fptr, infile, &status);
    printfitserror(status);

    // Get dimension
    naxis = 3;
    naxes = RALLOC(long, naxis);
    naxes[0] = h_zyx[1];
    naxes[1] = h_zyx[2];
    naxes[2] = h_zyx[0];
    ncoords = naxes[0] * naxes[1];

    // Explicitly create new image
    fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, &status);
    printfitserror(status);

    // Convert header keyrecords to FITS.
    nkeyrec = 0;
    end = false;
    k = 80 * 6;
    while(true){
        strncpy(keyrec, h_header + k, 80);
        k += 80;
        ++nkeyrec;

        // An END keyrecord was read, break while.
        if (strncmp(keyrec, "END       ", 10) == 0)
            end = true;

        // Ignore meta-comments (copyright information, etc.)
        if (keyrec[0] == '#')
            continue;

        // Strip off the newline
        i = strlen(keyrec) - 1;
        if (keyrec[i] == '\n')
            keyrec[i] = '\0';
                  
        fits_write_record(fptr, keyrec, &status);
        printfitserror(status);
                                    
        if(end)
            break;
    }

    // Get weighted data cube
    i = 0;
    for(k=0; k < naxes[0]; ++k){
        for(j=0; j < naxes[1]; ++j){
            // cout << h_datacube[i] << endl;// testing...
            h_datacube[0][i] /= h_weightscube[0][i];
            ++i;
        }
    }

    // Write image data to the output FITS file
    first = 1;
    fits_write_img(fptr, TDOUBLE, first, ncoords, h_datacube[0], &status);
    printfitserror(status);

    // Release
    DEALLOC(naxes);
    fits_close_file(fptr, &status);
    printfitserror(status);
}

/* Write ordered input map. */
void write_ordered_map(const char *infile, const char *outfile) {
    // Open original FITS file
    fitsfile *fptr;
    // int status;
    int status = 0; /* MUST initialize status */
    fits_open_file(&fptr, infile, READONLY, &status);
    printfitserror(status);

    // Read the dimension
    long naxis;
    fits_read_key_lng(fptr, "NAXIS", &naxis, NULL, &status);
    printfitserror(status);
    long *naxes = RALLOC(long, naxis);
    int nfound;
    fits_read_keys_lng(fptr, "NAXIS", 1, naxis, naxes, &nfound, &status);
    printfitserror(status);
    UTIL_ASSERT(nfound==naxis, "nfound!=naxis");
    long y = (uint32_t) naxes[0];
    long x = (uint32_t) naxes[1];
    long z = (uint32_t) (naxis == 3) ? naxes[2] : 1;

    // Create ordered input FITS file, delete any pre-existing file
    fitsfile *fptr1;
    fits_create_file(&fptr1, outfile, &status);
    printfitserror(status);

    // Copy primary header from original to ordered
    fits_copy_header(fptr, fptr1, &status);
    printfitserror(status);

    // Write image data for ordered
    fits_write_img(fptr1, TDOUBLE, 1, y*x*z, h_data, &status);
    printfitserror(status);

    // Copy secondary header from original to ordered
    fits_movabs_hdu(fptr, 2, NULL, &status);
    printfitserror(status);
    fits_copy_header(fptr, fptr1, &status);
    printfitserror(status);

    // Write image coordinates for ordered
    fits_write_col(fptr1, TDOUBLE, 1, 1, 1, y*x, h_lons, &status);
    printfitserror(status);
    fits_write_col(fptr1, TDOUBLE, 2, 1, y*x+1, y*x, h_lats, &status);
    printfitserror(status);

    // Close FITS files
    fits_close_file(fptr, &status);
    printfitserror(status);
    fits_close_file(fptr1, &status);
    printfitserror(status);

    // Release
    DEALLOC(naxes);
}

/* Print out cfitsio error messages */
static void printfitserror (int status){
    if (status==0)
        return;
    fits_report_error(stderr, status);  // print error report
    UTIL_FAIL("FITS error");
}
