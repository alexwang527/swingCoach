from display import DisplayImgManager
import os


class Main():

    def __init__(self):
        self.imageDisplay = DisplayImgManager()

    def img_estimation(self, img_path, img_path2, img_path3, img_path4, img_path5, img_path6, img_path7, img_path8 ):
        self.imageDisplay.estimate_img(img_path, img_path2, img_path3, img_path4, img_path5, img_path6, img_path7, img_path8 )

if __name__ == "__main__":
    app = Main()
    app.img_estimation("images/amsetup.jpg", "images/prosetup.jpg", 
                        "images/amtopswing.jpg", "images/protopswing.jpg",
                        "images/amimpact.jpg", "images/proimpact.jpg",
                        "images/amfinish.jpg", "images/profinish.jpg") 