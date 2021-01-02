from flask import Flask, render_template, url_for, request, redirect
from caption_func import *
import warnings
warnings.filterwarnings("ignore")

import webbrowser

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')





	
@app.route('/submit', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		img = request.files['userfile']

		# print(img)
		# print(img.filename)

		img.save("static/"+img.filename)

	
		caption = caption_this_image("static/"+img.filename)



		print(caption)
		url = "https://www.google.com.tr/search?q={}&tbm=isch".format(caption)
    	
		webbrowser.open_new_tab(url)
		result_dic = {
			'image' : "static/" + img.filename,
			'description' : caption
		}
	return render_template('index.html', results = result_dic)



if __name__ == '__main__':
	app.run(threaded=False)