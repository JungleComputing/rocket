#include <stdio.h>
#include <stdint.h>
#include <setjmp.h>

#include "jerror.h"
#include "jpeglib.h"

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

METHODDEF(void)
my_error_exit(j_common_ptr cinfo) {
  my_error_mgr* myerr = (my_error_mgr*) cinfo->err;

  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

static void ignore(j_decompress_ptr cinfo) {
   (void) cinfo; 
}

static boolean fill_input_buffer (j_decompress_ptr cinfo) {
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    return true;
}
static void skip_input_data (j_decompress_ptr cinfo, long num_bytes) {
    struct jpeg_source_mgr* src = (struct jpeg_source_mgr*) cinfo->src;

    if (num_bytes > 0) {
        src->next_input_byte += (size_t) num_bytes;
        src->bytes_in_buffer -= (size_t) num_bytes;
    }
}

// derived from https://stackoverflow.com/questions/5280756/libjpeg-ver-6b-jpeg-stdio-src-vs-jpeg-mem-src
void jpeg_mem_src(j_decompress_ptr cinfo, const void* buffer, long capacity) {
    struct jpeg_source_mgr* src;

    if (cinfo->src == NULL) {   /* first time for this JPEG object? */
        cinfo->src = (struct jpeg_source_mgr *)
            (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
            sizeof (struct jpeg_source_mgr));
    }

    src = (struct jpeg_source_mgr*) cinfo->src;
    src->init_source = ignore;
    src->fill_input_buffer = fill_input_buffer;
    src->skip_input_data = skip_input_data;
    src->resync_to_restart = jpeg_resync_to_restart; /* use default method */
    src->term_source = ignore;
    src->bytes_in_buffer = capacity;
    src->next_input_byte = (JOCTET*)buffer;
}


extern "C" int readJPEG (
        const char *input, 
        int32_t input_size, 
        int32_t *output_width,
        int32_t *output_height,
        char *output, 
        int32_t output_capacity
)
{
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
  int row_stride;
  JSAMPROW row;
  JDIMENSION i;

  /* Step 1: allocate and initialize JPEG decompression object */
  cinfo.err = jpeg_std_error((jpeg_error_mgr*) &jerr);
  jerr.pub.error_exit = my_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    goto err;
  }

  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */
  jpeg_mem_src(&cinfo, input, (long) input_size);

  /* Step 3: read file parameters with jpeg_read_header() */
  (void) jpeg_read_header(&cinfo, TRUE);

  /* Step 4: set parameters for decompression */
  // nothing here

  /* Step 5: Start decompressor */
  (void) jpeg_start_decompress(&cinfo);

  // Only support RGB using 3 components
  if (cinfo.out_color_space != JCS_RGB || cinfo.output_components != 3) {
      fprintf(stderr, "error: image not in RGB color space\n");
      goto err;
  }

  if (cinfo.output_width * cinfo.output_height * cinfo.output_components > output_capacity) {
      fprintf(stderr, "error: insufficient buffer space provided for image\n");
      goto err;
  }

  /* JSAMPLEs per row in output buffer */
  *output_width = (int) cinfo.output_width;
  *output_height = (int) cinfo.output_height;

  row_stride = cinfo.output_width * cinfo.output_components;
  row = (JSAMPROW) output;

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */
  while (cinfo.output_scanline < cinfo.output_height) {
    i = jpeg_read_scanlines(&cinfo, &row, 1);
    row += i * row_stride;
  }

  /* Step 7: Finish decompression */
  (void) jpeg_finish_decompress(&cinfo);

  /* Step 8: Release JPEG decompression object */
  jpeg_destroy_decompress(&cinfo);

  return 0;

err:
  // print error
  (*(cinfo.err->output_message))((j_common_ptr) &cinfo);
  jpeg_destroy_decompress(&cinfo);
  return 1;
}

