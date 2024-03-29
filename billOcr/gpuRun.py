import cv2
from pytesseract import Output

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import json
import re
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    # Enabling device placement logging causes any Tensor allocations or operations to be printed
    tf.debugging.set_log_device_placement(True)
    # If developing on a system with a single GPU, you can simulate multiple GPUs with virtual devices. This enables
    # easy testing of multi-GPU setups without requiring additional resources.
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=500),
         tf.config.LogicalDeviceConfiguration(memory_limit=500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    # Once there are multiple logical GPUs available to the runtime, you can utilize the multiple GPUs with
    # tf.distribute.Strategy
    strategy = tf.distribute.MirroredStrategy(logical_gpus)
    with strategy.scope():
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

        img = cv2.imread('bill.jpg')
        # convert image data to a dictionary
        txtdict = pytesseract.image_to_data(img, output_type=Output.DICT)
        #txtdict = pytesseract.image_to_data(Image.open('bill.jpg'), output_type=Output.DICT)
        # convert list elements to tensors in order to run the operation on GPU
        billInfo = []
        n = 0
        for tensor in txtdict['text']:
            tf.constant(tensor)
            billInfo.insert(n, tensor)
            n = n + 1

        # print(type(txtdict['text']))
        # print(txtdict['text'].count(""))

        # remove in necessary characters
        for i in billInfo:
            if "" in billInfo:
                billInfo.remove("")
            if " " in billInfo:
                billInfo.remove(" ")
            if "   " in billInfo:
                billInfo.remove("   ")
        # print(txtdict['text'].count(""))
        # print(txtdict['text'])

        # find the locations of the prices
        n = 0
        prices = []
        priceLocations = []
        for i in billInfo:
            if (re.search("[0-9]+(\.|\,)[0-9]{2}", i)):
                priceLocations.insert(n, n)
                prices.insert(n, i)
            n = n + 1

        # print(priceLocations)
        # print(prices)

        # prepare a python object for the bill data
        bill = {}
        bill["Description"] = ' '.join(billInfo[0:billInfo.index('Table:') - 1])
        bill["Table"] = billInfo.__getitem__(billInfo.index('Table:') + 1)
        bill["Staff"] = billInfo.__getitem__(billInfo.index('Staff') + 2)
        bill["Consumed Items"] = []
        bill["Consumed Items"].insert(0, {"Item": billInfo.__getitem__(priceLocations[0] - 1), "Price": prices[0]})

        for loc in range(1, len(priceLocations)):
            bill["Consumed Items"].insert(loc, {
                "Item": ' '.join(billInfo[priceLocations[loc - 1] + 1:priceLocations[loc] - 1]),
                "Price": prices[loc]})

        # convert into JSON:
        billJson = json.dumps(bill)

        # the result is a JSON string:
        print(billJson)

  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

