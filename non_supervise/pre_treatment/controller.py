import cv2

def launch_filter(img,path):
  img2= img.copy()
  if path == 1:
    img2 = filter(img2)
  
def test_img(img, img_filter, path, num):
  img2 = img.copy()
  mse = mse(img, img_filter)
  
def pipeline(img)
   num = 0 
   
   while(num != -1):
      img_filter = launch_filter(img, path)
      test_img(img,img_filter, path, num)
    
    
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse
