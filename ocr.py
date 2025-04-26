import timeit # returns time in seconds
from onnx.onnx_cpp2py_export import ONNX_ML
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from chain import reviser_chain
from dotenv import load_dotenv
load_dotenv()
from my_utils import chunk_string
import torch

use_gpu = False

def ocr(file_to_process):
    # for full script runtime
    start_time_full_script = timeit.default_timer()

    if torch.cuda.is_available() and use_gpu==True:
        # device = torch.device("cuda:0")
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # load model
    print("-------------------------------------------------")
    print("Load model...")
    start_time_load_model = timeit.default_timer()
    ###
    model = ocr_predictor('fast_base','crnn_mobilenet_v3_small', pretrained=True).to(device)
    ###
    elapsed = timeit.default_timer() - start_time_load_model
    print("runtime to load model: ", elapsed)

    # load pdf
    print("-------------------------------------------------")
    print("Load doc ...")
    doc = DocumentFile.from_images(file_to_process)

    # Analyze
    print("-------------------------------------------------")
    print("Run OCR...")
    start_time_run_model = timeit.default_timer()
    ###
    result = model(doc)
    ###
    elapsed = timeit.default_timer() - start_time_run_model
    print("runtime to run model: ", elapsed)

    # https://github.com/mindee/notebooks/blob/main/doctr/quicktour.ipynb
    print("-------------------------------------------------")
    print("Print result...")
    string_result = result.render()
    # print(string_result)

    # Runtime
    print("-------------------------------------------------")
    elapsed = timeit.default_timer() - start_time_full_script
    print("runtime of full script: ", elapsed)
    return string_result