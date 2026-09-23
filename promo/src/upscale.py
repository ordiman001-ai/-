import ncnn, numpy as np, sys, glob, os, time
from PIL import Image
T="/tmp/claude-0/-home-user--/803224e4-c7b3-564a-9a74-0c2f64db4169/scratchpad/"
net=ncnn.Net(); net.opt.use_vulkan_compute=False; net.opt.num_threads=4
net.load_param(T+"tools/re/models/realesrgan-x4plus.param"); net.load_model(T+"tools/re/models/realesrgan-x4plus.bin")
def run(tile):
    ex=net.create_extractor()
    t=np.ascontiguousarray(tile); m=ncnn.Mat.from_pixels(t, ncnn.Mat.PixelType.PIXEL_RGB, t.shape[1], t.shape[0])
    m.substract_mean_normalize([],[1/255.]*3); ex.input("data", m)
    _,out=ex.extract("output"); return np.array(out).transpose(1,2,0)
TS,P=192,12
for f in sorted(glob.glob("/home/user/-/reference/*.jpg")):
    name=os.path.basename(f)[:-4]; o=T+"up/"+name+".png"
    if os.path.exists(o): continue
    im=Image.open(f).convert("RGB"); w,h=im.size; C=48
    a=np.asarray(im.crop((C,C,w-C,h-C)))
    H,W,_=a.shape; out=np.zeros((H*4,W*4,3),np.float32); t0=time.time()
    for y in range(0,H,TS):
        for x in range(0,W,TS):
            y0,x0=max(0,y-P),max(0,x-P); y1,x1=min(H,y+TS+P),min(W,x+TS+P)
            r=run(a[y0:y1,x0:x1])
            ty1,tx1=min(H,y+TS),min(W,x+TS)
            out[y*4:ty1*4,x*4:tx1*4]=r[(y-y0)*4:(ty1-y0)*4,(x-x0)*4:(tx1-x0)*4]
    Image.fromarray((np.clip(out,0,1)*255+0.5).astype(np.uint8)).save(o)
    print(name,W*4,H*4,f"{time.time()-t0:.0f}s",flush=True)
