
import cv2
from pytesseract import Output

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import json
import re

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# convert image data to a dictionary
txtdict = pytesseract.image_to_data(Image.open('bill.jpg'), output_type=Output.DICT)
#print(type(txtdict['text']))
#print(txtdict['text'].count(""))

# remove in necessary characters
for i in txtdict['text']:
    if "" in txtdict['text']:
        txtdict['text'].remove("")
    if " " in txtdict['text']:
        txtdict['text'].remove(" ")
    if "   " in txtdict['text']:
        txtdict['text'].remove("   ")
#print(txtdict['text'].count(""))
#print(txtdict['text'])


# find the locations of the prices
n = 0
prices = []
priceLocations = []
for i in txtdict['text']:
    if (re.search("[0-9]+(\.|\,)[0-9]{2}", i)):
        priceLocations.insert(n, n)
        prices.insert(n, i)
    n = n + 1

#print(priceLocations)
#print(prices)

# prepare a python object for the bill data
bill = {}
bill["Description"] = ' '.join(txtdict['text'][0:txtdict['text'].index('Table:') - 1])
bill["Table"] = txtdict['text'].__getitem__(txtdict['text'].index('Table:') + 1)
bill["Staff"] = txtdict['text'].__getitem__(txtdict['text'].index('Staff') + 2)
bill["Consumed Items"] = []
bill["Consumed Items"].insert(0, {"Item": txtdict['text'].__getitem__(priceLocations[0] - 1), "Price": prices[0]})

for loc in range(1, len(priceLocations)):
    bill["Consumed Items"].insert(loc, {
        "Item": ' '.join(txtdict['text'][priceLocations[loc - 1] + 1:priceLocations[loc] - 1]),
        "Price": prices[loc]})

# convert into JSON:
billJson = json.dumps(bill)

# the result is a JSON string:
print(billJson)
