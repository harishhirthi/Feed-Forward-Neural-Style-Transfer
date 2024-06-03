import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
from models.Transform_net import TransFormerNet
import utils as utils

UPLOAD_FOLDER = os.path.join('static', 'uploads') # Folder to save the uploaded input image
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
ALLOWED_EXTENSIONS = {'jpg'}
app = Flask(__name__)
app.config['UPLOAD'] = UPLOAD_FOLDER

binaries_path = os.path.join('models', 'binaries')
img_save_path = os.path.join('static', 'output_images') # Folder to save the stylized image
Inference_config = dict()
Inference_config['save_folder'] = img_save_path
os.makedirs(img_save_path, exist_ok = True)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_default_device() -> None:
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))
        return torch.device('cuda')
    else:
        return torch.device('cpu')

"""Function to stylize the image and save it in a folder"""
def stylize(Inference_config, model_name):
    device = get_default_device()

    """Initializing Transformer model that stylizes the image"""
    style_net = TransFormerNet().to(device)
    trained_state = torch.load(os.path.join(binaries_path, model_name))
    binary = trained_state['state_dict']
    style_net.load_state_dict(binary, strict = True)
    style_net.eval()

    with torch.no_grad():
        content_img_path = Inference_config['image_path']
        content_img = utils.process_img(content_img_path, target_shape = 700)
        content_img = content_img.to(device)
        stylized_img = style_net(content_img).detach().cpu().numpy().squeeze(0)
        utils.save_and_display(Inference_config, stylized_img)

"""Function to get a input image and show the stylized image"""
@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        model_name = request.form.get('Modelname') # Used to get the model name.
        if file and allowed_file(file.filename): 
            filename = secure_filename(file.filename)
            Inference_config['content_img_name'] = filename
            Inference_config['image_path'] = os.path.join(app.config['UPLOAD'], filename)
            file.save(os.path.join(app.config['UPLOAD'], filename))
            stylize(Inference_config, model_name)
            image = os.path.join(Inference_config['save_folder'], f"Stylized-image-{filename.split('.')[0]}.jpg")
            return render_template('render.html', image = image)
    return render_template('render.html')

if __name__ == '__main__':
    app.run(debug = True)



