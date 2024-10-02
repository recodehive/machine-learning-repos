import scipy
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import sys
import shutil
from data_loader import DataLoader
import numpy as np
import os
import tensorflow as tf
import glob
import argparse
from preprocessing import ImageSlicer
from PIL import Image, ImageEnhance
parser = argparse.ArgumentParser()
parser.add_argument("--rem", '-r', help="delete files in directories saves and images",
                    action="store_true")
parser.add_argument("--pred",'-p',type=str, help="only run predict script")
args = parser.parse_args()
if args.rem:
    print("Deleting files...")
    files = glob.glob('../saves/*')
    for file_0 in files:
        shutil.rmtree(file_0)
    files = glob.glob('../images/unlabeled2017/*')
    for file_1 in files:
        os.remove(file_1)
class SRGAN(keras.Model):
    def __init__(self):
        super(SRGAN, self).__init__()
        self.channels = 3
        self.lr_height = 200
        self.lr_width = 200
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*2
        self.hr_width = self.lr_width*2
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.n_residual_blocks = 16
        optimizer = tf.optimizers.Adam(0.0002, 0.5)
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.im_path = '../../datasets/unlabeled2017/000000002272.jpg'

        # Configure data loader
        self.dataset_name = 'unlabeled2017'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = False
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        #self.generator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

        # High res. and low res. images
        img_hr = keras.Input(shape=self.hr_shape)
        img_lr = keras.Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = keras.Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = keras.applications.VGG19(weights="imagenet",include_top=False, input_shape=(self.hr_height,self.hr_width,self.channels))
        vgg.outputs = [vgg.layers[9].output]

        img = keras.Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return keras.Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = keras.layers.Activation('relu')(d)
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = keras.layers.UpSampling2D(size=2)(layer_input)
            u = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = keras.layers.Activation('relu')(u)
            return u
        # Low resolution image input
        img_lr = keras.Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = keras.layers.Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = keras.layers.Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = keras.layers.BatchNormalization(momentum=0.9)(c2)
        c2 = keras.layers.Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        #u2 = deconv2d(u1)
        # Generate high resolution output
        gen_hr = keras.layers.Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u1)

        return keras.Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = keras.Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = keras.layers.Dense(self.df*16)(d8)
        d10 = keras.layers.LeakyReLU(alpha=0.2)(d9)
        validity = keras.layers.Dense(1, activation='sigmoid')(d10)

        return keras.Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        for epoch in range(epochs):

            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            fake_hr = self.generator.predict(imgs_lr)
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                save_dir = ('../saves/' +str(epoch))
                os.mkdir(save_dir)
                model_save_dir = (save_dir + "/model.h5")
                #tf.saved_model.save(self.generator, save_dir)
                self.generator.save(model_save_dir)

    def sample_images(self, epoch):
        os.makedirs('../images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("../images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('../images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()
    def pred_images(self, im_path):
        files = '../saves/'
        save_paths = []
        for file in glob.glob("../saves/*"):
            try:
                save_paths.append(int(file[9:]))
            except ValueError:
                pass

        save_paths = max(save_paths)
        model_path = ('../saves/%s/model.h5' % (save_paths))
        self.generator = keras.models.load_model(model_path)
        
        os.makedirs('../images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2
        imgs_hr, imgs_lr = self.data_loader.load_pred(im_path)
        fake_hr = self.generator.predict(imgs_lr)
        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("../images/%s.png" % ("prediction"))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('../images/%s_lowres%d.png' % ("predicion", i))
            plt.close()

    def batch_image(self, path):
        filesToRemove = [os.path.join('temp/',f) for f in os.listdir('temp/')]
        for f in filesToRemove:
            os.remove(f)
        self.im_path = path
        files = '../saves/'
        save_paths = []
        for file in glob.glob("../saves/*"):
            try:
                save_paths.append(int(file[9:]))
            except ValueError:
                pass
        save_paths = max(save_paths)
        model_path = ('../saves/%s/model.h5' % (save_paths))
        self.generator = keras.models.load_model(model_path)
        self.generator.compile(loss='mse',
                              optimizer=tf.optimizers.Adam(0.0002, 0.5))
        self.slicer = ImageSlicer(path, (200,200),BATCH=False, PADDING=False)
        self.transformed_image = self.slicer.transform()
        self.batch_images = self.slicer.save_images(self.transformed_image)
        fake_img_batch = []
        for r in range(0,self.slicer.r*self.slicer.c):
            img = self.batch_images[r,:,:,:]
            img = np.expand_dims(img, axis=0)
            fake_img = self.generator.predict(img)
            fake_img = np.array(fake_img)
            fake_img = np.squeeze(fake_img,axis=0)
            fake_img = np.array(((fake_img + 0.2)*200), dtype=np.uint8)

            Image.fromarray(fake_img.astype(np.uint8)).save("temp/img%s.jpg" % r)
            fake_img_batch.append(fake_img)
        img_max_width = self.slicer.c*400
        img_max_height = self.slicer.r*400

        fake_img_batch = np.squeeze(fake_img_batch,axis=1)
        fake_img_batch = np.array(((fake_img_batch + 1)*127.5), dtype=int)
        fake_img_batch = np.reshape(fake_img_batch, (img_max_height,img_max_width,3),order='A')
        ims = os.listdir('temp')
        big_im = Image.new('RGB', (img_max_width,img_max_height))
        yy =0
        xx=0
        xy=0
        while xx != self.slicer.r+1:
            im = Image.open("temp/%s"%ims[xy])
            xy+=1

            if xx == self.slicer.r:
                big_im.paste(im, (xx*400,yy))
                if yy < self.slicer.c:
                    yy+=400
                    xx=0
                else:
                    break
            elif xx < self.slicer.r:
                big_im.paste(im, (xx*400,yy))
                xx+=1
            if xy == self.slicer.r*self.slicer.c:
                break
        enhancer_object = ImageEnhance.Contrast(big_im)
        out = enhancer_object.enhance(1.3)
        out.save('highres_output.jpg')
if __name__ == '__main__':
    gan = SRGAN()
    if args.pred is not None:
        gan.batch_image(args.pred)
    else:
        gan.train(epochs=30000, batch_size=1, sample_interval=50)
