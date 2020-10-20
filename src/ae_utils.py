import os
import sys
import src.params as params

def check_dirs():
    if not os.path.exists(params.RESULTS_DIR):
        try:
            os.mkdir(params.RESULTS_DIR)
        except:
            sys.exit('ERROR: Cannot create directory for results.')

    if params.PRINT_ENCODING == True:
        if not os.path.exists(params.ENCODING_DIR):
            try:
                os.mkdir(params.ENCODING_DIR)
            except:
                sys.exit('ERROR: Cannot create directory for the encoding of the binding sites.')
