import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline

def translation(prompt, trans_model):
    prompt = [prompt]
   
    tokenizer = MarianTokenizer.from_pretrained(trans_model)
    model = MarianMTModel.from_pretrained(trans_model)
   
    translated = model.generate(**tokenizer(prompt, return_tensors="pt", padding=True))
    #extra_translated = model.generate(**tokenizer(extra_prompt, return_tensors="pt", padding=True))

    for t in translated:
        text = tokenizer.decode(t, skip_special_tokens=True)

    #for t in extra_translated:
    #    extra_text = tokenizer.decode(t, skip_special_tokens=True)
       
    #print(text)
    return text
   
   
def generate(main_text, extra_text, negative_prompt, sd_model, file_name):
    device = "cuda:0"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model, torch_dtype=torch.float32)
    pipe.to(device)
   
    prompt = main_text + ',' + extra_text
    negative_prompt = negative_prompt
   
    generator = torch.Generator(device=device).manual_seed(589824)
    image = pipe(prompt + extra_prompt,
                 negative_prompt=negative_prompt,
                 height=768, width=768,
                 num_inference_steps=20,
                 guidance_scale=7.5,
                 generator=generator).images[0]
         
    save_path = './outputs/SD_result'
    image.save(save_path + '/' + file_name + '.png')

   
   
def parse_config():
    parser = argparse.ArgumentParser(description='stable diffusion unClip argment')
    parser.add_argument('--trans_model', type=str, default="Helsinki-NLP/opus-mt-ko-en", required=False, help='Enter your tranlation model. Default model is opus-mt-ko-en v2020')
    parser.add_argument('--sd_model', type=str, default = "friedrichor/stable-diffusion-2-1-realistic", required=False, help='Enter your stable diffusion model. Default model is stable-diffusion-2-1-realistic')
    parser.add_argument('--file_name', type=str, default=None, required=True, help='Enter your file name')
    parser.add_argument('--main_prompt', type=str, default=None, required=True, help='Enter your main prompt')
    parser.add_argument('--main_type', type=str, default='ko', required=False, help = 'Enter your main prompt language type. Defult is ko(korean). If result is not satisfied, then enter en and enter your prompt with English')
    parser.add_argument('--extra_prompt', type=str, default='최고의 품질, 아름다운, 환상적인, 최고의, 상세한 얼굴', required=False, help='Enter your extra_prompt. Default is 최고의 품질, 아름다운, 환상적인, 최고의, 상세한 얼굴')
    parser.add_argument('--extra_type', type=str, default='ko', required=False, help= 'Enter your extra prompt language type. Defult is ko(korean). If result is not satisfied, then enter en and enter your prompt with English')
    parser.add_argument('--n_prompt', type=str, default = None, required=False, help='Enter your nagetive prompt which you want add. Default nagative_prompt is disfigured, bad, immature, 3d, b&w, painting, facing the camera, cartoon, anime, ugly, (aged, white beard, black skin, wrinkle:1.1), (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2), (facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs')
    parser.add_argument('--n_type', type=str, default='ko', required=False, help= 'Enter your nagetive prompt language type. Defult is ko(korean). If result is not satisfied, then enter en and enter your prompt with English')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_config()
    main_prompt = args.main_prompt
    extra_prompt = args.extra_prompt
    trans_model = args.trans_model
    sd_model = args.sd_model
    n_prompt = args.n_prompt
    file_name = args.file_name
    main_type = args.main_type
    extra_type = args.extra_type
    n_type = args.n_type
    n_add = args.n_prompt
   
    if sd_model == "friedrichor/stable-diffusion-2-1-realistic":
       negative_prompt = "disfigured, bad, immature, 3d, b&w, painting, facing the camera, cartoon, anime, ugly, (aged, white beard, black skin, wrinkle:1.1), (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2), (facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs"
    else:
         negative_prompt = "disfigured, bad, immature, 3d, b&w, facing the camera, ugly, (aged, white beard, black skin, wrinkle:1.1), (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2), (facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs"
       

    if main_type == 'ko':
       main_text = translation(main_prompt, trans_model)
    else:
         main_text = main_prompt
         
    if extra_type == 'ko':
       extra_text = translation(extra_prompt, trans_model)
    else:
         extra_text = extra_prompt
         
    if n_add is not None:
       if n_type == 'ko':
          n_add = translation(n_prompt, trans_model)
          n_text = negative_prompt + ',' + n_add
       else:
            n_add = n_prompt
            n_text = negative_prompt + ',' + n_add
         
    if n_add is None:
       n_text = negative_prompt
       
    print('main',main_text)
    print('extra', extra_text)
    print('n', n_text)
    generate(main_text, extra_text, n_text, sd_model, file_name)
