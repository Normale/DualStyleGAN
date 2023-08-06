import os
import json
import numpy as np
import torch
from kafka import KafkaConsumer
from util import save_image, load_image
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
import argparse 
import base64
import requests
from PIL import Image
from io import BytesIO

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--style", type=str, default='rpg2', help="target style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator-001100.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")
        self.parser.add_argument("--align_face", action="store_true", default=True,  help="apply face alignment to the content image")
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('Time elapsed: %.2f' % (end_time-start_time))
    return wrapper

@timer_decorator
def process_images(args, device, generator, encoder):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    

    with torch.no_grad():
        viz = []
        # load content image
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device)
        viz += [I]

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                z_plus_latent=True, return_z_plus_latent=True, resize=False)    
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        viz += [img_rec]

        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color:
            latent[:,7:18] = instyle[:,7:18]
        # extrinsic styte code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

        # load style image if it exists
        S = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
            S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
            viz += [S]

        # style transfer 
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator([instyle], exstyle, input_is_latent=False, z_plus_latent=True,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('Generate images successfully!')
    
    save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

    print('Save images successfully!')


if __name__ == "__main__":

    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    print('Load models successfully!')


    # Set up Kafka consumer
    topic_name = "style-transfer"
    bootstrap_servers = "localhost:19092"
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="my-group"
    )
    consumer.subscribe([topic_name])
    print("Consumer up and running!")

    base_url = "http://localhost:5000/"  # Replace this with the base URL of your backend server
    from pathlib import Path
    for msg in consumer:
        print("Received message", msg.value)
        message_json = msg.value
        if not message_json:
            continue

        # Parse the JSON string into a dictionary
        try:
            message_json = json.loads(message_json)
        except json.JSONDecodeError as e:
            print(f"Error while parsing JSON: {e}")
            continue

        # Read the selectedImage file path from the URL
        selected_image_url = message_json.get("selectedImage")
        if selected_image_url:
            # Convert the URL to the local file path using pathlib
            image_path = Path(selected_image_url.replace(base_url, "").replace("/", os.path.sep))
            local_image_path = Path(r"C:\Users\bartek\GitHub\bachelor\services\backend\storage") / image_path
            print(f"{image_path=}")
            print(f"{local_image_path=}")
            if local_image_path.exists():
                try:
                    selected_image = Image.open(local_image_path)
                except Exception as e:
                    print(f"Error while reading the image: {e}")
                    continue
            else:
                print(f"Image not found at {local_image_path}")
                continue
        else:
            print("No 'selectedImage' URL found in the message.")
            continue


        # Parse options from JSON message
        parser = TestOptions()
        args = parser.parse()
        for key, value in message_json.items():
            print(f"Setting {key} to {value}")
            setattr(args, key, value)
        print(args)
        process_images(args, device, generator, encoder)
        consumer.commit()