---
title: 生成式AI详解
abbrlink: 1
---
## Chapter 5. Paint

到目前为止，我们已经探索了各种训练模型生成新样本的方法，只需要给定我们想要模仿的训练数据集。我们将其应用于几个数据集，并看到在每种情况下，VAEs和GANs如何能够学习潜在空间和原始像素空间之间的映射。通过从潜空间的分布中采样，我们可以使用生成模型将该向量映射到像素空间中的新图像。

请注意，到目前为止，我们看到的所有示例都是从头开始生成新的观察结果的，也就是说，除了从潜在空间中采样的随机潜在向量用于生成图像外，没有其他输入。生成模型的另一个应用是在风格迁移领域。本文的目标是建立一个模型，可以转换输入的基图像，以便给人一种它与给定的一组风格图像来自同一集合的印象。这种技术有明显的商业应用，现在被用于计算机图形软件、计算机游戏设计和移动电话应用程序。图5-1展示了其中的一些例子。

![image-20230418193555408](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230418193555408.png)

在本章中，你将学习如何构建两种不同类型的风格迁移模型(CycleGAN和Neural style transfer)，并将这些技术应用于你自己的照片和艺术品。我们将从参观一家水果和蔬菜店开始，那里的一切并不像看上去那样……

### CycleGAN

风格迁移常用的模型:循环一致对抗网络，或CycleGAN。原始论文代表了风格迁移领域的重要一步，因为它展示了如何训练一个模型，在没有成对样本的训练集的情况下，将风格从参考图像集复制到不同的图像上。之前的风格迁移模型，如pix2pix，要求训练集中的每个图像都存在于源域和目标域。虽然对于某些风格的问题设置(例如，黑白到彩色照片，映射到卫星图像)，可以制造这种数据集，但对于其他问题，这是不可能的。例如，我们没有莫奈画《睡莲》系列的池塘的原始照片，也没有毕加索的帝国大厦画作。将马和斑马站在相同位置的照片进行整理也需要花费巨大的精力。

CycleGAN论文在pix2pix论文发布几个月后发布，展示了如何训练一个模型来解决源域和目标域没有图像对的问题。图5-4分别展示了pix2pix和CycleGAN的成对数据集和未成对数据集之间的差异。

![image-20230418194309402](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230418194309402.png)

虽然pix2pix只能在一个方向上工作(从源到目标)，但CycleGAN同时在两个方向上训练模型，以便模型学习将图像从目标到源以及源到目标。这是模型架构的结果，因此您可以免费获得相反的方向。

现在让我们看看如何在Keras中构建CycleGAN模型。首先，我们将使用前面的苹果和橘子的例子来遍历CycleGAN的每个部分，并对该架构进行实验。然后，我们将应用相同的技术来建立一个模型，可以将给定艺术家的风格应用到您选择的照片中。

#### Data

使用apple2orange数据集

https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip

#### Overview

CycleGAN实际上由四个模型、两个发生器和两个鉴别器组成。第一个生成器G_AB将图像从域A转换到域B。第二个生成器G_BA将图像从域B转换到域A。

由于我们没有成对的图像来训练我们的生成器，我们还需要训练两个判别器来确定生成器生成的图像是否令人信服。第一个鉴别器d_A被训练成能够识别来自域A的真实图像和由生成器G_BA产生的假图像之间的差异。相反，鉴别器d_B被训练成能够识别来自域B的真实图像和由生成器G_AB产生的假图像之间的差异。四种型号的对应关系如图5-5所示。

![image-20230419170224638](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230419170224638.png)

让我们首先看一下生成器的架构。通常，4个CycleGAN生成器采用两种形式之一:U-Net或ResNet(残差网络)。在他们早期的pix2pix论文中，作者使用了U-Net架构，但他们为CycleGAN切换到了ResNet架构。本章将从U-Net开始构建这两种架构

#### The Generators (U-Net)

![image-20230419170753927](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230419170753927.png)

与变分自编码器类似，U-Net由两部分组成:下采样的一半，输入图像在空间上进行压缩，但在通道上进行扩展，上采样的一半，在空间上扩展表示，同时减少通道数量。

然而，与VAE不同的是，在网络的上采样和下采样部分中，形状相同的层之间也存在跳跃连接。VAE是线性的;数据通过网络从输入流到输出，一层接一层。U-Net是不同的，因为它包含跳过连接，允许信息通过网络的部分快捷方式传递到后面的层。这里的直觉是，随着网络的下采样部分的每一层的后续，模型越来越多地捕获图像的内容，并丢失关于哪里的信息。在U的顶点，特征图将学习对图像中内容的上下文理解，而对其位置的理解很少。对于预测分类模型，这就是我们所需要的，因此我们可以将其连接到最终的密集层，以输出图像中特定类存在的概率。然而，对于原始的U-Net应用程序(图像分割)和风格迁移，至关重要的是，当我们上采样回原始图像大小时，我们将下采样期间丢失的空间信息传递回每一层。这正是我们需要skip连接的原因。它们允许网络将下采样过程中捕获的高级抽象信息(即图像风格)与从网络中的前一层反馈回来的特定空间信息(即图像内容)相融合。

 为了构建跳跃连接，我们需要引入一种新类型的层:Concatenate。

##### CONCATENATE LAYER

Concatenate层只是沿着特定的轴(默认是最后一个轴)将一组层连接在一起。例如，在Keras中，我们可以将前面的x层和y层连接在一起，如下所示:

```python
Concatenate()([x,y])
```

在U-Net中，我们使用级联层将上采样层连接到网络的下采样部分中同等大小的层。这些层沿着通道维度连接在一起，因此通道的数量从k增加到2k，而空间维度的数量保持不变。注意，级联层中不需要学习权重;它们只是用来“粘合”前几层。

 生成器还包含另一个新的层类型，

InstanceNormalization。

##### INSTANCE NORMALIZATION LAYER

这个CycleGAN的生成器使用实例规范化层而不是批规范化层，这在风格迁移问题中可以导致更令人满意的结果。

实例规范化层(instancnormalization layer)单独规范化每个观测值，而不是作为一个批处理进行规范化。与BatchNormalization层不同，它不需要在训练期间将mu和sigma参数作为运行平均值进行计算，因为在测试时该层可以像在训练时一样对每个实例进行归一化。用于归一化每层的均值和标准偏差计算每个通道和每个观测。

此外，对于该网络中的实例归一化层，没有权重需要学习，因为我们没有使用缩放(gamma)或移位(beta)参数。

图5-7展示了批量归一化和实例归一化以及其他两种归一化方法(层归一化和组归一化)之间的区别。

 ![image-20230419172836165](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230419172836165.png)

这里，N是批处理轴，C是通道轴，(H, W)表示空间轴。因此，立方体表示归一化层的输入张量。蓝色像素使用相同的均值和方差(根据这些像素的值计算)进行归一化。

##### core code

keras版本

```python
# 核心部分代码
    def unet_generator(self):
        '''Unet生成器网络'''
        def downsample(layer_input, filters, kernel_size=4):
            '''下采样'''
            d = Conv2D(filters, kernel_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = ReLU()(d)
            return d

        def upsample(layer_input, skip_input, filters, kernel_size=4, droput_rate=0):
            '''上采样'''
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = ReLU()(u)
            if droput_rate:
                u = Dropout(droput_rate)(u)
            u = Concatenate()([u, skip_input])
            return u

        # image input
        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = downsample(img, self.gen_n_filters)
        d2 = downsample(d1, self.gen_n_filters * 2)
        d3 = downsample(d2, self.gen_n_filters * 4)
        d4 = downsample(d3, self.gen_n_filters * 8)

        # Upsampling
        u1 = upsample(d4, d3, self.gen_n_filters * 4)
        u2 = upsample(u1, d2, self.gen_n_filters * 2)
        u3 = upsample(u2, d1, self.gen_n_filters)
        u4 = UpSampling2D(size=2)(u3)

        output = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output)

```

#### The Discriminators

到目前为止，我们看到的鉴别器只输出一个数字:输入图像为“实”的预测概率。我们将构建的CycleGAN中的鉴别器输出8 × 8单通道张量，而不是单个数字。

这样做的原因是CycleGAN继承了PatchGAN模型的鉴别器架构，在PatchGAN模型中，鉴别器将图像划分为重叠的方形“补丁”，并猜测每个补丁是真的还是假的，而不是对整个图像进行预测。因此鉴别器的输出是一个张量，其中包含每个patch的预测概率，而不仅仅是一个数字。

请注意，当我们通过网络传递图像时，这些补丁是同时预测的——我们不会手动分割图像，然后逐个通过网络传递每个补丁。由于鉴别器的卷积结构，将图像划分为小块是很自然的。

使用PatchGAN鉴别器的好处是，损失函数可以衡量鉴别器根据风格而不是内容区分图像的能力。由于鉴别器预测的每个单独元素仅基于图像的一个小正方形，因此它必须使用补丁的样式而不是其内容来做出决定。这正是我们所需要的;我们宁愿我们的鉴别器擅长于识别两张图片在风格上的不同，而不是内容上的不同。

##### core code

鉴别器的Keras代码

 

```python
    def discriminator(self):
        '''判别器'''

        def conv4(layer_input, filters, stride=2, norm=True):
            y = Conv2D(filters, kernel_size=4, strides=stride, padding='same')(layer_input)
            if norm:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = LeakyReLU(0.2)(y)
            return y

        img = Input(shape=self.img_shape)

        y = conv4(img, self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, self.disc_n_filters * 2, stride=2)
        y = conv4(y, self.disc_n_filters * 4, stride=2)
        y = conv4(y, self.disc_n_filters * 8, stride=1)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same')(y)

        return Model(img, output)
```

CycleGAN鉴别器是一系列卷积层，所有层都具有实例规范化(第一层除外)。

最后一层是卷积层，只有一个滤波器，没有激活。

#### Compiling the CycleGAN

总结一下，我们的目标是建立一组模型，可以将域a(例如苹果的图像)转换为域B(例如橘子的图像)，反之亦然。因此，我们需要编译四个不同的模型，两个生成器和两个鉴别器，如下所示:

g_AB学习将图像从域A转换到域B。

g_BA学习将图像从域B转换到域A。

d_A学习来自域A的真实图像和g_BA生成的假图像之间的差异。

d_B学习来自域B的真实图像和g_AB生成的假图像之间的差异。

我们可以直接编译这两个鉴别器，因为我们有输入(来自每个域的图像)和输出(二进制响应:1表示图像来自该域，0表示它是生成的假图像)。

```python
# 编译判别器
self.d_A = self.discriminator()
self.d_B = self.discriminator()
self.d_A.compile(loss='mse',
				 optimizer=Adam(self.learning_rate, 0.5),
       			 metrics=['accuracy'])
self.d_B.compile(loss='mse',
                 optimizer=Adam(self.learning_rate, 0.5),
                 metrics=['accuracy'])
```

 但是，我们不能直接编译生成器，因为我们的数据集中没有成对的图像。相反，我们根据三个标准同时判断生成器:

+ 1.有效性。每个生成器产生的图像是否欺骗了相关的鉴别器?(例如，输出是否来自g_BA fool d_A，输出是否来自g_AB fool d_B?)

+ 2.重建。如果我们依次应用两个生成器(在两个方向上)，我们会返回原始图像吗?CycleGAN得名于这个循环重构准则。

+ 3.身份。如果我们将每个生成器应用于其自己的目标域的图像，图像是否保持不变?

下面展示了如何编译一个模型来满足这三个条件(代码中的数字标记对应前面的列表)。

```python
# 编译生成器
self.g_AB = self.unet_generator()
self.g_BA = self.unet_generator()
# For the combined model we will only train the generators
self.d_A.trainable = False
self.d_B.trainable = False
img_A = Input(shape=self.img_shape)
img_B = Input(shape=self.img_shape)
fake_A = self.g_BA(img_B)
fake_B = self.g_AB(img_A)

valid_A = self.d_A(fake_A)
valid_B = self.d_B(fake_B)

reconstr_A = self.g_BA(fake_B)
reconstr_B = self.g_AB(fake_A)

img_A_id = self.g_BA(img_A)
img_B_id = self.g_AB(img_B)

self.combined = Model(inputs=[img_A, img_B],
                      output=[valid_A, valid_B,
                              reconstr_A, reconstr_B,
                              img_A_id, img_B_id])
self.combined.compile(loss=['mse', 'mse',
                            'mse', 'mse',
                            'mse', 'mse'
                            ],
                      loss_weights=[
                          self.lambda_validation,
                          self.lambda_validation,
                          self.lambda_reconstr,
                          self.lambda_reconstr,
                          self.lambda_id,
                          self.lambda_id,
                      ],
                      optimizer=self.optimizer
                      )
self.d_A.trainable = True
self.d_B.trainable = True
```

 组合模型接受来自每个域的一批图像作为输入，并为每个域提供三个输出(以匹配三个标准)——因此，总共有六个输出。请注意我们如何冻结判别器中的权重，这是典型的GANs，以便组合模型只训练生成器权重，即使判别器涉及到模型中。

总损失是每个准则损失的加权和。均方误差用于有效性标准——根据真实(1)或虚假(0)响应检查鉴别器的输出——平均绝对误差用于基于图像到图像的标准(重建和一致性)。

#### Training the CycleGAN

 在我们的判别器和组合模型编译后，我们现在可以训练我们的模型了。这遵循了标准的GAN实践，即交替训练鉴别器和训练生成器(通过组合模型)。

 

```python
def train_discriminators(self, imgs_A, imgs_B, valid, fake):

    # Translate images to opposite domain
    fake_B = self.g_AB.predict(imgs_A)
    fake_A = self.g_BA.predict(imgs_B)

    self.buffer_B.append(fake_B)
    self.buffer_A.append(fake_A)

    fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A)))
    fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))

    # Train the discriminators (original images = real / translated = Fake)
    dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
    dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

    dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
    dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    # Total disciminator loss
    d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

    return (
        d_loss_total[0]
        , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
        , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
        , d_loss_total[1]
        , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
        , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
    )

def train_generators(self, imgs_A, imgs_B, valid):

    # Train the generators
    return self.combined.train_on_batch([imgs_A, imgs_B],
                                        [valid, valid,
                                         imgs_A, imgs_B,
                                         imgs_A, imgs_B])

def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + self.disc_patch)
    fake = np.zeros((batch_size,) + self.disc_patch)

    for epoch in range(self.epoch, epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch()):

            d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
            g_loss = self.train_generators(imgs_A, imgs_B, valid)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                  % (self.epoch, epochs,
                     batch_i, data_loader.n_batches,
                     d_loss[0], 100 * d_loss[7],
                     g_loss[0],
                     np.sum(g_loss[1:3]),
                     np.sum(g_loss[3:5]),
                     np.sum(g_loss[5:7]),
                     elapsed_time))

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                self.sample_images(data_loader, batch_i, run_folder, test_A_file, test_B_file)
                self.combined.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (self.epoch)))
                self.combined.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

        self.epoch += 1
```

1.我们对真实图像使用1响应，对生成图像使用0响应。注意每个patch有一个响应，因为我们使用的是PatchGAN鉴别器。

2.为了训练鉴别器，我们首先使用各自的生成器创建一批假图像，然后我们在这个假集和一批真实图像上训练每个鉴别器。通常，对于CycleGAN，批处理大小为1(单个图像)。

3.通过前面编译的组合模型，在一个步骤中一起训练生成器。看看这六个输出如何与前面在编译期间定义的六个损失函数相匹配。

#### Analysis of the CycleGAN

让我们看看CycleGAN在简单数据集apples and oranges上的表现，并观察改变损失函数中的权重参数会如何对结果产生巨大影响。

现在您已经熟悉了CycleGAN架构，您可能会意识到这个图像代表了判断组合模型的三个标准:有效性、重建和身份。

让我们用代码库中的适当函数重新标记这个图像，以便更清楚地看到这一点(如图5-8所示)。

![image-20230423153642240](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423153642240.png)

我们可以看到网络的训练是成功的，因为每个生成器都明显地改变了输入图像，使其看起来更像来自相反域的有效图像。此外，当一个接一个地应用生成器时，输入图像和重建图像之间的差异最小。最后，当每个生成器应用于自己输入域的图像时，图像不会发生显著变化。

在最初的CycleGAN论文中，除了必要的重构损失和有效性损失外，身份损失是可选的。为了证明身份项在损失函数中的重要性，让我们通过在损失函数中设置身份损失权重参数为零，来看看如果我们去掉身份项会发生什么(图5-9)。

![image-20230423154002808](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423154002808.png)

CycleGAN仍然成功地将橘子翻译成苹果，但装橘子的托盘的颜色已经从黑色变成了白色，因为现在没有身份损失术语来防止这种背景颜色的变化。身份项有助于调节生成器，以确保它只调整完成转换所需的图像部分，而不是更多。

这突出了确保三个损失函数的权重很好地平衡的重要性——身份损失过少，会出现颜色偏移问题;身份丢失太多，CycleGAN没有足够的动力来改变输入，使其看起来像来自相反域的图像。

#### Creating a CycleGAN to Paint Like Monet

现在我们已经探索了CycleGAN的基本结构，我们可以将注意力转向更有趣和令人印象深刻的技术应用。

在最初的CycleGAN论文中，一个突出的成就是模型能够学习如何将给定的照片转换成特定艺术家风格的绘画。由于这是一个CycleGAN，该模型也能够以另一种方式转换，将艺术家的绘画转换为逼真的照片。

要下载从monet到照片的数据集：

https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip

#### The Generators (ResNet)

在这个例子中，我们将介绍一种新型的生成器架构:残差网络(residual network，简称ResNet)。ResNet架构类似于U-Net，因为它允许来自网络中先前层的信息提前跳过一层或多层。然而，ResNet不是通过将网络的下采样部分连接到相应的上采样层来创建U形，而是由相互堆叠的残差块构建，其中每个块包含一个跳跃连接，在将其传递到下一层之前，对块的输入和输出进行求和。单个残差块如图5-10所示。

![image-20230423155141774](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423155141774.png)

在我们的CycleGAN中，图中的“权重层”是卷积的具有实例规范化的层。在Keras中，残差块的编码方法如例5-8所示。

```python
from keras.layers.merge import add
def residual(layer_input, filters):
    shortcut = layer_input
    y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(layer_input)
    y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(y)
    y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
    return add([shortcut, y])
```

在残差块的两侧，我们的ResNet生成器还包含下采样和上采样层。ResNet的总体架构如图5-11所示。

![image-20230423155831146](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423155831146.png)

研究表明，ResNet架构可以被训练到数百甚至数千层的深度，并且不会受到梯度消失问题的影响，其中早期层的梯度很小，因此训练非常缓慢。这是因为误差梯度可以通过作为残差块的一部分的跳跃连接在网络中自由反向传播。此外，人们相信，添加额外的层绝不会导致模型精度的下降，因为跳跃连接确保了，如果不能提取进一步的信息特征，则总是可以通过前一层的身份映射。

####  Analysis of the CycleGAN

在原始的CycleGAN论文中，该模型被训练了200个epoch，以实现艺术家到照片风格迁移的最先进结果。在图5-12中，我们显示了每个生成器在早期训练过程不同阶段的输出，以显示模型开始学习如何将莫奈的绘画转换为照片和反之亦然时的进展。

在上面一行中，我们可以看到，莫奈所使用的独特的颜色和笔触逐渐转化为照片中所期望的更自然的颜色和平滑的边缘。类似地，下面一行的情况正好相反，因为生成器学会了如何将一张照片转换成莫奈可能自己画的场景。

 ![image-20230423160148468](C:\Users\cheris\AppData\Roaming\Typora\typora-user-images\image-20230423160148468.png)

图5-13是原始论文中该模型经过200次epoch训练后得到的部分结果。

![image-20230423160303136](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423160303136.png)

### Neural Style Transfer

到目前为止，我们已经看到了CycleGAN如何在两个域之间转换图像，其中训练集中的图像不一定是成对的。现在我们来看看风格转移的另一种应用，我们根本没有训练集，而是希望将一张图像的风格转移到另一张图像上，如图5-14所示。这被称为神经风格转移。

![image-20230423162133259](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423162133259.png)

这个想法的前提是我们要最小化一个损失函数，该函数是三个不同部分的加权和:

内容损失：我们希望组合图像包含与基础图像相同的内容。

风格损失：我们希望组合图像与风格图像具有相同的总体风格。

总方差损失：我们希望组合后的图像看起来平滑而不是像素化。

我们通过梯度下降来最小化这种损失，也就是说，在多次迭代中，我们按与损失函数的负梯度成比例的数量更新每个像素值。这样，损失随着每次迭代逐渐减少，我们最终得到一个图像，该图像将一个图像的内容与另一个图像的风格相融合。

 通过梯度下降优化生成的输出与我们迄今为止解决生成模型问题的方式不同。之前，我们通过在整个网络中反向传播误差来训练一个深度神经网络，如VAE或GAN，以从训练数据集中学习，并将学到的信息泛化以生成新图像。在这里，我们不能采用这种方法，因为我们只有两个图像要处理，基础图像和样式图像。然而，正如我们将看到的，我们仍然可以使用预训练的深度神经网络来提供损失函数中每个图像的重要信息。我们将从定义三个单独的损失函数开始，因为它们是神经风格迁移引擎的核心。

##### Content Loss

 内容损失衡量了两幅图像在主题和内容的整体位置方面的差异。两幅包含相似场景的图像(例如，一排建筑物的照片和另一幅从不同光线、不同角度拍摄的相同建筑物的照片)的损失应该小于两幅包含完全不同场景的图像。简单地比较两幅图像的像素值是不行的，因为即使是在同一场景的两幅不同的图像中，我们也不会期望单个像素值相似。我们真的不希望内容损失关心单个像素的值;我们更希望它根据高层特征(如建筑物、天空或河流)的存在和大致位置对图像进行评分。

我们以前见过这个概念。这是深度学习的整个前提——训练用于识别图像内容的神经网络，通过结合前一层的简单特征，自然地在网络的更深层学习更高级别的特征。因此，我们需要的是一个已经成功训练过识别图像内容的深度神经网络，这样我们就可以利用网络的深层来提取给定输入图像的高层特征。如果我们测量基础图像的输出和当前组合图像之间的均方误差，我们就有了内容损失函数!

我们将使用的预训练网络称为VGG19。这是一个19层的卷积神经网络，经过训练可以将ImageNet数据集中的100多万张图像分类为1000个对象类别。组网图如图5-15所示。

 

![image-20230423162835600](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423162835600.png)

```python
from keras.applications import vgg19
from keras import backend as K

base_image_path = '/path/base_image.jpg'
style_reference_image_path = '/path/style_image.jpg'

content_weight = 0.01
base_image = K.variable(preprocess_image(base_image_path)) 
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0) 
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False) 
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
layer_features = outputs_dict['block5_conv2'] 
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :] 
def content_loss(content, gen):
    return K.sum(K.square(gen - content))
content_loss = content_weight * content_loss(base_image_features
                                           , combination_features)
```

注意：

Keras库包含可以导入的预训练的VGG19模型。

我们定义两个Keras变量来保存基础图像和样式图像，以及一个占位符，它将包含生成的组合图像。

VGG19模型的输入张量是三个图像的连接。

这里，我们创建了VGG19模型的一个实例，指定了输入张量和我们想要预加载的权重。include_top = False参数指定我们不需要为最终用于图像分类的网络密集层加载权重。这是因为我们只对前面的卷积层感兴趣，这些层捕获输入图像的高级特征，而不是原始模型被训练为输出的实际概率。

我们用来计算内容损失的层是第五个块的第二层卷积层。选择网络中较浅或较深点的层会影响损失函数如何定义“内容”，因此会改变生成的组合图像的属性。在这里，我们从输入张量中提取基础图像特征和组合图像特征，这些特征已经通过VGG19网络输入。

内容损失是两个图像的选定层的输出之间的平方和乘以一个加权参数。

##### Style Loss

风格损失更难量化——我们如何衡量两个图像之间的风格相似性?神经风格迁移论文中给出的解决方案是基于这样的想法:**在给定层中，风格相似的图像通常具有相同的特征图之间的相关性模式**。通过一个例子我们可以更清楚地看到这一点。假设在VGG19网络中，我们有一些层，其中一个通道已经学会识别图像的绿色部分，另一个通道已经学会识别尖峰，另一个已经学会识别图像的棕色部分。来自这些通道的三个输入的输出(特征图)如图5-16所示。

 ![image-20230423170631065](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423170631065.png)

我们可以看到A和B在样式上很相似——都是绿草。我们可以把特征图展平，然后**计算点积。如果结果值很高，则特征图高度相关;如果该值较低，则特征图不相关。**我们可以定义一个矩阵，其中包含层中所有可能特征对之间的点积。这被称为Gram矩阵。图5-17展示了每个图像的三个特征的Gram矩阵。

 ![image-20230423172710390](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423172710390.png)

很明显，样式相似的图像A和B在这一层具有相似的Gram矩阵。即使它们的内容可能非常不同，但Gram矩阵(度量层中所有特征对之间的相关性)是相似的。因此，为了计算风格损失，我们需要做的就是为整个网络中的一组层计算基图像和组合图像的Gram矩阵(GM)，并使用误差平方和比较它们的相似性。从代数上讲，对于大小为M(高度x宽度)、有N个通道的给定层(l)，基础图像(S)和生成图像(G)之间的风格损失可以写成:
$$
{\cal L}_{G M}\left(S,G,l\right)=\frac{1}{4N_{l}^{2}M_{l}^{2}}\sum_{i j}\left(G M\left[l\right](S)_{i j}-G M\left[l\right](G)_{i j}\right)^{2}
$$
注意它是如何根据通道数量(N)和层大小(M)进行缩放的。这是因为我们将整体风格损失计算为几个层的加权和，所有这些层都有不同的大小。总风格损失计算如下:
$$
{\cal L}_{s t y l e}\left(S,{\cal G}\right)=\sum_{l=0}^{L}w_{l}{\cal L}_{G M}\left(S,{\cal G},l\right)
$$

```python
style_loss = 0.0
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1'] 
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :] 
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    style_loss += (style_weight / len(feature_layers)) * sl
    
# 样式损失是在五个层上计算的——VGG19模型的五个块中的每一个的第一个卷积层。

# 在这里，我们从通过VGG19网络提供的输入张量中提取风格图像特征和组合图像特征。

# 样式损失由加权参数和计算的层数来缩放
```



##### Total Variance Loss

总方差损失只是组合图像中噪声的度量。为了判断图像的噪声程度，我们可以将其向右移动一个像素，然后计算平移后图像与原始图像之间的平方和。为了平衡，我们也可以执行相同的过程，但将图像向下移动一个像素。这两项的和就是总方差损失。

 

```python
def total_variation_loss(x):
    a = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :]) 
    b = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :]) 
    return K.sum(K.pow(a + b, 1.25))
tv_loss = total_variation_weight * total_variation_loss(combination_image) 
16

loss = content_loss + style_loss + tv_loss  # 总损失
# 图像和同一图像之间的平方差将向下移动一个像素。图像和同一图像之间的平方差向右移动了一个像素。总方差损失由一个加权参数缩放。总体损失是内容、风格和总方差损失的总和。

 
```

##### Running the Neural Style Transfer

该过程以基图像作为起始合并图像进行初始化。

在每次迭代时，我们将当前合并的图像(变平)传递给scipy中的优化函数fmin_l_bfgs_b优化包，根据L- BFGS-B算法执行一个梯度下降步骤。

在这里，evaluator是一个对象，包含计算前面描述的总体损失以及相对于输入图像的损失梯度的方法。

```python
from scipy.optimize import fmin_l_bfgs_b
iterations = 1000
x = preprocess_image(base_image_path) 
for i in range(iterations):
    x, min_val, info = fmin_l_bfgs_b( 
        evaluator.loss 
        , x.flatten()
        , fprime=evaluator.grads 
        , maxfun=20
        )
```

##### Analysis of the Neural Style Transfer Model

图5-18显示了学习过程中三个不同阶段的神经风格迁移过程输出，参数如下:

+ content_weight: 1
+ style_weight: 100
+ total_variation_weight: 20

![image-20230423175534545](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230423175534545.png)

我们可以看到，随着每个训练步骤的进行，该算法在风格上越来越接近风格图像，并失去了基础图像的细节，同时保留了整体内容结构。有许多方法可以试验这种架构。你可以尝试改变损失函数或用于确定内容相似度的层中的权重参数，看看这是如何影响组合输出图像和训练速度的。您还可以尝试衰减风格损失函数中赋予每个层的权重，以使模型偏向于迁移更精细或更粗糙的风格特征。

###  Summary

在本章中，我们探索了两种不同的生成新艺术品的方法:CycleGAN和神经风格迁移。

CycleGAN方法允许我们训练一个模型来学习艺术家的一般风格，并将其转移到照片上，以生成看起来就像艺术家画了照片中的场景一样的输出。该模型还免费为我们提供了反向过程，将绘画转换为逼真的照片。至关重要的是，来自每个域的配对图像不需要CycleGAN工作，使其成为一种极其强大和灵活的技术。

神经风格迁移技术允许我们将单个图像的风格迁移到基图像上，使用巧妙选择的损失函数，惩罚模型偏离基础图像的内容和风格图像的艺术风格太远，同时保持输出的平滑程度。这项技术已经被许多知名的应用程序商业化，可以将用户的照片与一组特定风格的绘画融合在一起。在下一章中，我们将从基于图像的生成建模转移到一个新的挑战领域:基于文本的生成建模。

 

## Chapter 6. Write

在本章中，我们将探讨在文本数据上构建生成模型的方法。文本数据和图像数据之间有几个关键的区别，这意味着许多适用于图像数据的方法并不那么容易适用于文本数据。特别是:

+ 文本数据由离散的块(字符或单词)组成，而图像中的像素是连续光谱中的点。我们可以很容易地让绿色像素更蓝色，但如何让单词cat更像单词dog就不是很明显了。这意味着我们可以轻松地将反向传播应用于图像数据，因为我们可以计算损失函数相对于单个像素的梯度，以确定像素颜色应改变的方向，以最小化损失。对于离散的文本数据，我们不能像通常那样应用反向传播，因此我们需要找到一种解决这个问题的方法。

+ 文本数据有时间维度但没有空间维度，而图像数据有两个空间维度但没有时间维度。单词的顺序在文本数据中非常重要，单词倒过来是没有意义的，而图像通常可以翻转而不影响内容。此外，模型需要捕获单词之间的长期顺序依赖关系:例如，问题的答案或延续代词的上下文。对于图像数据，可以同时处理所有像素。
+ 文本数据对单个单位(单词或字符)的微小变化非常敏感。图像数据通常对单个像素单位的变化不太敏感——即使改变了一些像素，一幅房子的图像仍然可以被识别为一所房子。然而，对于文本数据，即使改变几个单词，也可能彻底改变文章的含义，或使其变得毫无意义。这使得训练一个模型来生成连贯的文本变得非常困难，因为每个单词对文章的整体含义都至关重要。
+ 文本数据具有基于规则的语法结构，而图像数据不遵循关于如何分配像素值的规则。例如，在任何内容中写“The cat sat on The having”都没有语法意义。还有一些语义规则很难建模;说“I am in the beach”是没有意义的，尽管从语法上讲，这句话没有任何错误。

 文本建模已经取得了良好的进展，但上述问题的解决方案仍然是正在进行的研究领域。我们将从最常用和已建立的用于生成序列数据(如文本)的模型之一开始，递归神经网络(RNN)，特别是长短期记忆(LSTM)层。本章还将探索一些在问题-答案对生成领域取得有希望成果的新技术。



###  Long Short-Term Memory Networks

LSTM网络是一种特殊类型的循环神经网络(RNN)。rnn包含一个循环层(或单元格)，它能够通过使自己在特定时间步的输出成为下一个时间步输入的一部分来处理顺序数据，以便过去的信息可以影响当前时间步的预测。我们说LSTM网络是指具有LSTM递归层的神经网络。

当首次引入rnn时，递归层非常简单，仅由tanh算子组成，该算子确保时间步之间传递的信息在-1和1之间缩放。然而，这被证明会受到梯度消失问题的影响，并且不能很好地扩展到长序列数据。

1997年，Sepp Hochreiter和Jürgen Schmidhuber在一篇论文中首次介绍了LSTM单元。在这篇论文中，作者描述了lstm如何没有普通rnn所经历的梯度消失问题，并且可以在数百个时间步长的序列上进行训练。从那时起，LSTM架构已经被调整和改进，像门控循环单元(gru)这样的变体现在被广泛使用并作为Keras中的层。让我们首先看看如何使用Keras构建一个非常简单的LSTM网络，它可以生成伊索寓言风格的文本。

### Your First LSTM Network

11339 aesop

数据集：http://www.gutenberg.org/cache/epub/11339/pg11339.txt

#### Tokenization

第一步是对文本进行清理和标记化。分词(Tokenization)是将文本分割为单个单元(如单词或字符)的过程。

如何标记文本将取决于您试图使用文本生成模型实现的目标。同时使用单词标记和字符标记各有利弊，您的选择将影响您在建模和模型输出之前需要如何清理文本。

+ 如果使用单词标记:

所有文本都可以转换为小写，以确保句子开头大写的单词与出现在句子中间的相同单词的标记化方式相同。

但在某些情况下，这可能不是我们想要的;例如，一些专有名词，如名称或地点，可以保持大写，以便独立标记它们。

文本词汇表(训练集中不同的单词的集合)可能非常大，其中一些单词出现得非常稀疏，或者可能只出现一次。将稀疏单词替换为未知单词的标记可能是明智的，而不是将它们作为单独的标记包括在内，以减少神经网络需要学习的权重数量。

单词可以被词根化，这意味着它们被简化为最简单的形式，这样一个动词的不同时态保持在一起。例如，browse(浏览)、browsing(浏览)、browses(浏览)和browsing(浏览)都可以用眉毛来表示。您需要将标点符号标记化，或者完全删除它。使用单词标记化意味着模型将永远无法预测训练词汇表之外的单词。

+ 如果你使用字符标记:

模型可能会在训练词汇表之外生成形成新单词的字符序列——这在某些上下文中可能是可取的，但在另一些上下文中则不是。大写字母可以转换为对应的小写字母，也可以保留为单独的标记。使用字符分词时，词汇表通常要小得多。这有利于模型训练速度，因为最终输出层需要学习的权重更少。

在本例中，我们将使用小写单词分词，而不使用单词词干提取。我们还将标记标点符号，例如，我们希望模型能够预测何时应该结束句子或开始/结束演讲标记。最后，我们将用一个新的故事角色块(||||||||||||||||||||)替换故事之间的多个换行符。这样，当我们使用模型生成文本时，我们可以用这个字符块填充模型，这样模型就知道要从头开始一个新的故事。

```python
import re

from keras.preprocessing.text import Tokenizer
filename = "./data/aesop/data.txt"
with open(filename, encoding='utf-8-sig') as f:
    text = f.read()
seq_length = 20
start_story = '| ' * seq_length

# CLEANUP
text = text.lower()
text = start_story + text
text = text.replace('\n\n\n\n\n', start_story)
text = text.replace('\n', ' ')
text = re.sub('  +', '. ', text).strip()
text = text.replace('..', '.')
text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
text = re.sub('\s{2,}', ' ', text)

# TOKENIZATION
tokenizer = Tokenizer(char_level = False, filters = '')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]
```

![image-20230424144605303](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424144605303.png)

![image-20230424144629316](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424144629316.png)

#### Building the Dataset

我们的LSTM网络将被训练为预测序列中的下一个单词，给定此点之前的单词序列。例如，我们可以给模型输入贪心的猫和和的标记，并期望模型输出一个合适的下一个单词(例如，dog，而不是in)。

 

```python
import numpy as np
from keras.utils import np_utils

def generate_sequences(token_list, step):
    X = []
    y = []
    for i in range(0, len(token_list) - seq_length, step):
        X.append(token_list[i: i + seq_length])
        y.append(token_list[i + seq_length])
    y = np_utils.to_categorical(y, num_classes = total_words)
    num_seq = len(X)
    print('Number of sequences:', num_seq, "\n")
    return X, y, num_seq
step = 1
seq_length = 20
X, y, num_seq = generate_sequences(token_list, step)
X = np.array(X)
y = np.array(y)
```

#### The LSTM Architecture

整个模型的架构如图6-3所示。模型的输入是一个整数标记序列，输出是词汇表中每个单词在下一个序列中出现的概率。为了详细理解它是如何工作的，我们需要引入两种新的层类型，嵌入和LSTM。

 ![image-20230424153701138](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424153701138.png)

#### The Embedding Layer

嵌入层本质上是一个查找表，它将每个标记转换为长度为embedding_size的向量(图6-4)。因此，这一层学习到的权重数量等于词汇表的大小乘以embedding_size。

 

![image-20230424154117205](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424154117205.png)

输入层将形状为[batch_size, seq_length]的整数序列张量传递到嵌入层，嵌入层输出形状为[batch_size, seq_length, embedding_size]的张量。然后将其传递给LSTM层(图6-5)。

 

![image-20230424154234610](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424154234610.png)

我们将每个整数标记嵌入到一个连续向量中，因为它使模型能够学习每个单词的表示，并能够通过反向传播进行更新。我们也可以对每个输入标记进行独热编码，但首选使用嵌入层，因为它使嵌入本身可训练，从而使模型在决定如何嵌入每个标记以提高模型性能方面具有更大的灵活性。

####  The LSTM Layer

要理解LSTM层，我们必须首先了解一般的循环层是如何工作的。

循环层具有能够处理顺序输入数据$$[x_1,…,x_n]$$的特殊属性。它由一个单元格组成，当序列x的每个元素通过它时，它更新其隐藏状态$$h_t$$，每次一个时间步。隐藏状态是一个长度等于单元格中单元数量的向量，它可以被认为是单元格当前对序列的理解。在时间步t，单元格使用隐藏状态$$h_{t-1}$$的前一个值和当前时间步$$x_t$$的数据来生成更新的隐藏状态向量$$h_t$$。这个循环过程一直持续到序列的末尾。

一旦序列完成，该层输出单元的最终隐藏状态$$h_n$$，然后传递到网络的下一层。这个过程如图6-6所示。

 

![image-20230424155154532](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424155154532.png)

为了更详细地解释这一点，让我们展开这个过程，以便我们可以确切地看到单个序列是如何穿过该层的(图6-7)。

 ![image-20230424155759670](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424155759670.png)

在这里，我们通过在每个时间步长绘制单元的副本来表示循环过程，并显示隐藏状态在流过单元时如何不断更新。我们可以清楚地看到前一个隐藏状态是如何与当前序列数据点(即当前嵌入的词向量)混合以产生下一个隐藏状态的。在输入序列中的每个单词被处理后，该层的输出是单元的最终隐藏状态。重要的是要记住，此图中的所有单元格共享相同的权重(因为它们实际上是同一个单元格)。这个图和图6-6没有什么区别。这只是绘制循环层机制的另一种方式。

####  The LSTM Cell

现在我们已经看到了通用循环层的工作原理，让我们看看单个LSTM单元的内部。

单元的工作是输出一个新的隐藏状态$$h_t$$，给定它之前的隐藏状态$$h_{t-1}$$和当前的词嵌入$$x_t$$。总结一下，$$h_t$$的长度等于LSTM中的单元数。这是一个在定义层时设置的参数，与序列的长度无关。确保你没有混淆术语cell和unit。LSTM层中有一个单元，由它包含的单元数量定义，就像我们之前故事中的囚犯单元包含许多囚犯一样。我们经常将循环层绘制为展开的单元链，因为它有助于可视化隐藏状态在每个时间步是如何更新的。

一个LSTM细胞维护一个细胞状态$$C_t$$，这可以被认为是细胞关于序列当前状态的内部信念。这与隐藏状态$$h_t$$不同，$$h_t$$在最后一个时间步长之后最终由单元输出。单元状态的长度与隐藏状态相同(单元中的单元数)。

让我们更仔细地看一下单个单元格，以及隐藏状态是如何更新的(如图6-8所示)。

 ![image-20230424162101787](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424162101787.png)

```python
from keras.layers import Dense, LSTM, Input, Embedding, Dropout
from keras.models import Model
from keras.optimizers import RMSprop

n_units = 256
embedding_size = 100
text_in = Input(shape = (None,))
x = Embedding(total_words, embedding_size)(text_in)
x = LSTM(n_units)(x)

x = Dropout(0.2)(x)
text_out = Dense(total_words, activation = 'softmax')(x)
model = Model(text_in, text_out)
opti = RMSprop(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=opti)

epochs = 100
batch_size = 32
model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle = True)
```

### Generating New Text

现在我们已经编译和训练了LSTM网络，我们可以通过应用以下过程开始使用它来生成长字符串文本:

1. 向网络提供现有的单词序列，并要求它预测接下来的单词。

2. 将这个单词添加到现有序列中并重复。网络将输出我们可以从中采样的每个单词的概率集合。因此，我们可以使文本的生成是随机的，而不是确定性的。此外，我们可以在采样过程中引入一个温度参数，以表明我们希望该过程具有多大的确定性。

 

```python
def sample_with_temp(preds, temperature=1.0): 
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)
def generate_text(seed_text, next_words, model, max_sequence_len, temp):
    output_text = seed_text

    seed_text = start_story + seed_text 
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] 
        token_list = token_list[-max_sequence_len:] 
        token_list = np.reshape(token_list, (1, max_sequence_len))
        probs = model.predict(token_list, verbose=0)[0] 
        y_class = sample_with_temp(probs, temperature = temp) 
        output_word = tokenizer.index_word[y_class] if y_class > 0 else ''
        if output_word == "|": 
            break
        seed_text += output_word + ' ' 
        output_text += output_word + ' '
    return output_text
```

![image-20230424171249034](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424171249034.png)

关于这两篇文章有几点需要注意。首先，两者在风格上类似于原始训练集中的寓言。它们都以熟悉的故事中人物的陈述开始，通常带有言语标记的文本更像对话，使用人称代词，由所说的单词的出现而准备。

其次，与温度= 1.0时生成的文本相比，在温度= 0.2时生成的文本不那么冒险，但在选择单词方面更连贯，因为较低的温度值导致更确定性的采样。最后，很明显，这两种方法都不能很好地跨越多个句子，因为LSTM网络无法掌握它生成的单词的语义。为了生成语义合理的可能性更大的段落，我们可以构建一个人工辅助的文本生成器，其中模型输出概率最高的10个单词，然后最终由人类从这个列表中选择下一个单词。这类似于手机上的预测文本，你可以从已经输入的单词中选择几个单词。为了证明这一点，图6-10显示了具有最高概率的前10个单词遵循各种序列(不是来自训练集)。

 ![image-20230424171433039](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230424171433039.png)

该模型能够在一系列上下文中为下一个最有可能的单词生成合适的分布。例如，即使模型从未被告知词性，如名词、动词、形容词和介词，但它通常能够将单词分成这些类别，并以语法正确的方式使用它们。它还可以猜测，以鹰为故事开头的文章更可能是an，而不是a。图6-10中的标点符号示例表明，模型对输入序列的细微变化也很敏感。在第一段中(狮子说，)，模型以98%的可能性猜测语音标记紧随其后，因此从句在口语对话之前。然而，如果我们输入下一个单词作为and，它能够理解，现在不太可能有词性标记，因为从句更有可能取代对话，句子更有可能继续作为描述性散文。

 

### RNN Extensions

上一节中的网络是一个简单的例子，展示了如何训练LSTM网络以学习如何生成给定样式的文本。在本节中，我们将探讨对这一思想的几种扩展。

#### Stacked Recurrent Networks

我们刚刚看到的网络包含单个LSTM层，但我们也可以训练具有堆叠LSTM层的网络，这样就可以从文本中学习到更深的特征。

为了实现这一点，我们将第一个LSTM层中的return_sequences参数设置为True。这使得层从每个时间步输出隐藏状态，而不仅仅是最后一个时间步。然后，第二个LSTM层可以使用第一层的隐藏状态作为其输入数据。如图6-11所示，整体模型架构如图6-12所示。

 

![image-20230425091603960](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425091603960.png)

```python
text_in = Input(shape = (None,))
embedding = Embedding(total_words, embedding_size)
x = embedding(text_in)
x = LSTM(n_units, return_sequences = True)(x)
x = LSTM(n_units)(x)
x = Dropout(0.2)(x)
text_out = Dense(total_words, activation = 'softmax')(x)
model = Model(text_in, text_out)
```

#### Gated Recurrent Units

另一种常用的RNN层是门控循环单元(GRU)。与LSTM单元的关键区别如下:

1. 遗忘门和输入门被重置门和更新门取代。

2. 没有单元状态或输出门，只有从单元输出的隐藏状态。隐藏状态分4步更新，如图6-13所示。

 ![image-20230425092121683](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425092121683.png)

+ 前一个时间步的隐藏状态$$h_{t-1}$$和当前的词嵌入状态$$x_t$$被连接起来，用于创建重置门。该门是一个密集层，具有权重矩阵$$W_r$$和sigmoid激活函数。结果向量$$r_t$$的长度等于单元格中单元的数量，并存储在0到1之间的值，这些值决定了前一个隐藏状态$$h_{t-1}$$的多少信息应该被带入计算单元格的新信念。
+ 重置门应用于隐藏状态$$h_{t-1}$$，并与当前的词嵌入$$x_t$$连接。然后，该向量被馈送到具有权重矩阵W和tanh激活函数的密集层，以生成一个向量$$\tilde{h}_t$$，该向量存储细胞的新信念。它的长度等于单元格中的单元数，并存储在-1到1之间的值。
+ 前一个时间步的隐藏状态$$h_{t-1}$$和当前词嵌入$$x_t$$的连接也用于创建更新门。该门是一个密集层，具有权重矩阵$$W_z$$和sigmoid激活。所得向量z的长度等于单元格中单元的数目，存储在0和1之间的值，用于确定有多少新的深信$$\tilde{h}_t$$融合到当前的隐藏状态$$h_{t-1}$$中。
+ 细胞$$\tilde{h}_t$$的新信念与当前隐藏状态$$h_{t-1}$$按更新门$$z_t$$确定的比例混合，以产生更新后的隐藏状态$$h_t$$，该状态由细胞输出。

#### Bidirectional Cells

对于在推理时模型可以获得整个文本的预测问题，没有理由只向前处理序列——它可以向后处理。双向层通过存储两组隐藏状态来利用这一点:一组是在通常的正向处理序列时产生的，另一组是在向后处理序列时产生的。这样，该层可以从之前的和给定时间步后的信息中学习。

在Keras中，这是作为循环层的包装实现的，如下所示:

```
layer = Bidirectional(GRU(100))
```

结果层中的隐藏状态是长度等于被包装单元中单元数量的两倍的向量(连接前向和后向隐藏状态)。因此，在这个例子中，层的隐藏状态是长度为200的向量。

 

#### Encoder–Decoder Models

到目前为止，我们已经研究了使用LSTM网络来生成现有文本序列的延续。我们已经看到单个LSTM层如何顺序处理数据，以更新表示该层当前对序列的理解的隐藏状态。通过将最终的隐藏状态连接到密集层，网络可以输出下一个单词的概率分布。

对于某些任务，目标不是预测现有序列中的单个下一个单词;相反，我们希望预测一个完全不同的单词序列，该序列在某种程度上与输入序列相关。这种类型的任务的一些例子有:

+ 语言翻译网络：接收源语言的文本字符串，目标是输出翻译成目标语言的文本。
+ 问题生成网络：接收一段文本，目标是生成一个可以针对文本提出的可行问题。
+ 文本摘要网络：接收一段很长的文本，目标是对该文本进行简短的摘要。

 

对于这类问题，我们可以使用一种称为编码器-解码器的网络。我们已经在图像生成的背景下看到了一种编码器-解码器网络:变分自动编码器。对于顺序数据，编码器-解码器过程如下所示。

+ 编码器RNN将原始输入序列汇总为单个向量。
+ 这个向量用于初始化RNN解码器。
+ 解码器RNN在每个时间步的隐藏状态连接到一个密集层，该层输出单词词汇表的概率分布。

这样，解码器可以生成一个新的文本序列，它已经用编码器产生的输入数据的表示进行了初始化。这个过程如图6-14所示，以英语和德语之间的翻译为例。

 

![image-20230425101945430](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425101945430.png)

编码器的最终隐藏状态可以被认为是整个输入文档的表示。然后，解码器将这种表示转换为顺序输出，例如将文本翻译成另一种语言，或与文档有关的问题。

在训练过程中，将解码器在每个时间步产生的输出分布与真实的下一个单词进行比较，以计算损失。在训练过程中，解码器不需要从这些分布中采样来生成单词，因为后续的单元被提供的是下一个单词的真实值，而不是从前一个输出分布中采样的单词。这种训练编码器-解码器网络的方式被称为教师强迫。我们可以想象，网络是一个学生，有时会做出错误的分布预测，但无论网络在每个时间步长输出什么，老师都提供正确的响应作为网络尝试下一个单词的输入。

 

### A Question and Answer Generator

现在，我们将把所有内容放在一起，构建一个可以从文本块中生成问题和答案对的模型。这个项目的灵感来自qgen-workshop的TensorFlow代码库和Tong Wang, Xingdi Yuan和Adam Trischler提出的模型。该模型由两部分组成:

+ RNN从文本块中识别候选答案

+ 编码器-解码器网络，在RNN突出显示的候选答案之一的情况下，生成合适的问题。

例如，考虑以下关于足球比赛的文本段落的开头:

 

```tex
The winning goal was scored by 23-year-old striker Joe Bloggs during the match
between Arsenal and Barcelona .
Arsenal recently signed the striker for 50 million pounds . The next match is in
two weeks time, on July 31st 2005 . "
```

我们希望我们的第一个网络能够识别潜在的答案，例如:

 

```tex
"Joe Bloggs"
"Arsenal"
"Barcelona"
"50 million pounds"
"July 31st 2005"
```

我们的第二个网络应该能够在给定每个答案的情况下生成一个问题，例如:

 

```tex
"Who scored the winning goal?"
"Who won the match?"
"Who were Arsenal playing?"
"How much did the striker cost?"
"When is the next match?"
```

让我们首先看一下我们将更详细地使用的数据集。

#### A Question-Answer Dataset

我们将使用Maluuba NewsQA数据集，你可以按照GitHub上的说明下载它。得到的train.csv、test.csv和dev.csv文件应该放在图书仓库的./data/qa/文件夹中。这些文件都具有相同的列结构，如下所示:

+ story_id：故事的唯一标识符

+ story_text：(例如，“制胜球是由23岁的前锋乔·布洛格斯在比赛中打进的……”)。

+ question：(例如，“前锋花了多少钱?”)。

+ answer_token_ranges表示答案在故事文本中的标记位置(例如，24:27)。如果答案在文章中出现多次，可能会有多个区间(用逗号分隔)。

这些原始数据经过处理并标记化，以便能够将其用作我们模型的输入。经过这种转换后，训练集中的每个观察值由以下五个特征组成:

+ document_tokens分词后的故事文本(例如，[1,4633,7,66,11，…])，用零剪裁/填充，长度为max_document_length(一个参数)。
+ question_input_tokens分词后的问题(例如，[2,39,1,52，…])，填充0，长度为max_question_length(另一个参数)。
+ question_output_tokens分词后的问题，偏移一个时间步长(例如，[39,1,52,1866，…]，用零填充，长度为max_question_length。
+ answer_mask二进制掩码矩阵，形状为[max_answer_length, max_document_length]。如果问题的答案的第i个单词位于文档的第j个单词，则矩阵的[i, j]值为1，否则为0。
+ answer_labels长度为max_document_length的二进制向量(例如[0,1,1,0，…])。如果文档中的第i个单词被认为是答案的一部分，则向量的第i个元素为1，否则为0。

现在让我们看一下能够从给定的文本块中生成问题-答案对的模型架构。

 

#### Model Architecture

图6-15展示了我们将要构建的整体模型架构。如果这看起来很吓人，不要担心!它只是由我们已经见过的元素构建而成，我们将在本节中一步一步地介绍这个架构。

 ![image-20230425145311499](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425145311499.png)

让我们首先看一下构建图顶部模型部分的Keras代码，它预测文档中的每个单词是否是答案的一部分。

```python
from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, Lambda
from keras.models import Model, load_model
import keras.backend as K
from qgen.embedding import glove
#### PARAMETERS ####
VOCAB_SIZE = glove.shape[0] # 9984
EMBEDDING_DIMENS = glove.shape[1] # 100
GRU_UNITS = 100
DOC_SIZE = None
ANSWER_SIZE = None
Q_SIZE = None
document_tokens = Input(shape=(DOC_SIZE,), name="document_tokens") 
embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIMENS
 	, weights=[glove], mask_zero = True, name = 'embedding') 
document_emb = embedding(document_tokens)
answer_outputs = Bidirectional(GRU(GRU_UNITS, return_sequences=True)
 	, name = 'answer_outputs')(document__emb) 
answer_tags = Dense(2, activation = 'softmax'
    , name = 'answer_tags')(answer_outputs)
```

注意：

+ 文档令牌作为模型的输入。这里，我们使用变量DOC_SIZE来描述输入的大小，但该变量实际上被设置为None。这是因为模型的架构不依赖于输入序列的长度——层中的单元格数量将自适应等于输入序列的长度，因此我们不需要显式指定它。
+ 嵌入层用GloVe词向量初始化(在下面的侧边栏中解释)。
+ 循环层是一个双向的GRU，它在每个时间步返回隐藏状态。
+ 输出密集层在每个时间步与隐藏状态连接，只有两个单元，具有softmax激活，表示每个单词是答案一部分(1)或不是答案一部分(0)的概率。

 

##### GLOVE WORD VECTORS

嵌入层是用一组预训练的词嵌入初始化的，而不是我们之前看到的随机向量。这些词向量是Stanford GloVe(“全局向量”)项目的一部分，该项目使用无监督学习来获取大量单词的代表向量。

这些向量具有许多有益的特性，如连接词之间的向量相似度。例如，单词man和woman的嵌入向量与单词king和queen之间的向量大致相同。这就好像单词的性别被编码到单词向量存在的潜在空间中。用GloVe初始化嵌入层通常比从头开始训练要好，因为捕获单词表示的大量艰苦工作已经通过GloVe训练过程实现了。然后，您的算法可以调整单词嵌入以适应您的机器学习问题的特定上下文。

为了在这个项目中使用GloVe词向量，请从GloVe项目网站下载文件GloVe . 6b .100d.txt(60亿个单词，每个单词的嵌入长度为100)，然后从图书库中运行以下Python脚本，修剪该文件，使其只包含训练语料库中存在的单词:

 

```
python ./utils/write.py
```



模型的第二部分是编码器-解码器网络，它接受给定的答案，并尝试制定匹配的问题(图6-15的底部部分)。

 

```python
encoder_input_mask = Input(shape=(ANSWER_SIZE, DOC_SIZE)
 	, name="encoder_input_mask")
encoder_inputs = Lambda(lambda x: K.batch_dot(x[0], x[1])
 	, name="encoder_inputs")([encoder_input_mask, answer_outputs])
encoder_cell = GRU(2 * GRU_UNITS, name = 'encoder_cell')(encoder_inputs) 
decoder_inputs = Input(shape=(Q_SIZE,), name="decoder_inputs") 
decoder_emb = embedding(decoder_inputs) 
decoder_emb.trainable = False
decoder_cell = GRU(2 * GRU_UNITS, return_sequences = True, name = 'decoder_cell')
decoder_states = decoder_cell(decoder_emb, initial_state = [encoder_cell]) 
decoder_projection = Dense(VOCAB_SIZE, name = 'decoder_projection'
 	, activation = 'softmax', use_bias = False)
decoder_outputs = decoder_projection(decoder_states) 
total_model = Model([document_tokens, decoder_inputs, encoder_input_mask]
 	, [answer_tags, decoder_outputs])
answer_model = Model(document_tokens, [answer_tags])
decoder_initial_state_model = Model([document_tokens, encoder_input_mask]
 	, [encoder_cell])
```

注意：

+ 答案掩码作为输入传递给模型，这允许我们将隐藏状态从单个答案范围传递到编码器-解码器。这是通过Lambda层实现的。
+ 编码器是一个GRU层，它将给定答案范围的隐藏状态作为输入数据。
+ 解码器的输入数据是与给定答案范围匹配的问题。
+ 问题词标记通过与答案识别模型相同的嵌入层传递。
+ 解码器是一个GRU层，使用编码器的最终隐藏状态进行初始化。
+ 解码器的隐藏状态通过一个密集层传递，以生成序列中下一个单词的整个词汇表的分布。

这就完成了我们的问题-答案对生成网络。为了训练网络，我们分批传递文档文本、问题文本和答案掩码作为输入数据，并最小化答案位置预测和问题词生成的交叉熵损失，平均加权。

#### Inference

为了在一个它从未见过的输入文档序列上测试模型，我们需要运行以下过程:

+ 将文档字符串提供给答案生成器，以生成文档中答案的示例位置。
+ 选择这些答案块中的一个，将其传递到编码器-解码器问题生成器(即，创建适当的答案掩码)。
+ 将文档和答案掩码提供给编码器，以生成解码器的初始状态。
+ 用这个初始状态初始化解码器，并将&lt;START&gt;Token来生成问题的第一个单词。继续这个过程，一个一个输入生成的单词，直到&lt;END&gt;Token由模型预测。

如前所述，在训练过程中，模型使用教师强迫将基本真实单词(而不是预测的下一个单词)输入到解码器单元。然而，在推理过程中，模型必须自己生成一个问题，因此我们希望能够将预测的单词反馈给解码器单元，同时保留其隐藏状态。

实现这一目标的一种方法是定义一个额外的Keras模型(question_model)，该模型接受当前单词标记和当前解码器隐藏状态作为输入，并输出预测的下一个单词分布和更新的解码器隐藏状态。

 

```python
decoder_inputs_dynamic = Input(shape=(1,), name="decoder_inputs_dynamic")
decoder_emb_dynamic = embedding(decoder_inputs_dynamic)
decoder_init_state_dynamic = Input(shape=(2 * GRU_UNITS,)
 	, name = 'decoder_init_state_dynamic')
decoder_states_dynamic = decoder_cell(decoder_emb_dynamic
 	, initial_state = [decoder_init_state_dynamic])
decoder_outputs_dynamic = decoder_projection(decoder_states_dynamic)

question_model = Model([decoder_inputs_dynamic, decoder_init_state_dynamic]
    , [decoder_outputs_dynamic, decoder_states_dynamic])
```

然后，我们可以在循环中使用该模型，逐字生成输出问题。

```python
test_data_gen = test_data()
batch = next(test_data_gen)
answer_preds = answer_model.predict(batch["document_tokens"])
idx = 0
start_answer = 37
end_answer = 39
answers = [[0] * len(answer_preds[idx])]
for i in range(start_answer, end_answer + 1):
    answers[idx][i] = 1
answer_batch = expand_answers(batch, answers)
next_decoder_init_state = decoder_initial_state_model.predict(
 	[answer_batch['document_tokens'][[idx]], answer_batch['answer_masks'][[idx]]])
word_tokens = [START_TOKEN]
questions = [look_up_token(START_TOKEN)]
ended = False
while not ended:
    word_preds, next_decoder_init_state = question_model.predict(
 															[word_tokens, next_decoder_init_state])
    next_decoder_init_state = np.squeeze(next_decoder_init_state, axis = 1)
    word_tokens = np.argmax(word_preds, 2)[0]
    questions.append(look_up_token(word_tokens[0]))
    if word_tokens[0] == END_TOKEN:
        ended = True
questions = ' '.join(questions)
```

#### Model Results

模型的示例结果如图6-16所示。根据模型，右边的图表显示了文档中每个单词构成答案一部分的概率。然后将这些答案短语提供给问题生成器，该模型的输出显示在图的左侧(“预测问题”)。

首先，请注意答案生成器如何能够准确识别文档中哪些单词最有可能包含在答案中。这已经相当令人印象深刻了，因为它以前从未见过此文本，也可能从未见过文档中包含在答案中的某些单词，例如Bloggs。它能够从上下文中理解这很可能是一个人的姓氏，因此很可能构成答案的一部分。

 

![image-20230425154320301](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425154320301.png)

编码器从每个可能的答案中提取上下文，以便解码器能够生成合适的问题。值得注意的是，编码器能够捕捉到第一个答案中提到的人，23岁的前锋乔·布洛格斯，可能有一个与他的进球能力相关的匹配问题，并能够将此上下文传递给解码器，从而生成问题“谁得分了&lt;UNK&gt;?，而不是，例如，“谁是总统?”解码器已经用标签&lt;UNK&gt;完成了这个问题，但不是因为它不知道接下来要做什么——它预测接下来的单词很可能来自核心词汇表之外。

我们不应该惊讶于模型诉诸于使用标签&lt;UNK&gt;在这种情况下，由于原始语料库中的许多小众词将被标记为这种方式。我们可以看到，在每种情况下，解码器都根据答案的类型选择了正确的问题“类型”——谁、多少钱、什么时候问。但是仍然有一些问题，例如问他损失了多少钱?而不是为这名前锋支付了多少钱?这是可以理解的，因为解码器只有最终的编码器状态，而不能引用原始文档以获取额外的信息。

有一些对编码器-解码器网络的扩展，可以提高模型的准确性和生成能力。其中使用最广泛的两种是pointer networks和attention mechanisms。pointer networks使模型能够"指向"输入文本中的特定单词，以包括在生成的问题中，而不仅仅依赖于已知词汇表中的单词。这有助于解决&lt;UNK&gt;前面提到的问题。我们将在下一章中详细探讨注意力机制。

###  Summary

在本章中，我们看到了如何应用循环神经网络来生成模仿特定写作风格的文本序列，以及如何从给定的文档中生成合理的问题-答案对。

我们探索了两种不同类型的循环层，长短期记忆(long short-term memory, lstm)和GRU，并了解了这些单元如何堆叠或双向形成更复杂的网络架构。本章介绍的编码器-解码器架构是一个重要的生成工具，因为它允许序列数据被压缩为单个向量，然后可以解码为另一个序列。这适用于除问答对生成外的一系列问题，如翻译和文本摘要。

在这两种情况下，我们都看到了如何将非结构化的文本数据转换为可与循环神经网络层一起使用的结构化格式的重要性。很好地理解张量的形状在数据流经网络时如何变化，也是构建成功网络的关键，而递归层中需要特别注意顺序数据的时间维度，因为转换过程增加了额外的复杂性。在下一章中，我们将看到有多少关于rnn的相同思想可以应用于另一种类型的序列数据:音乐。

##  Chapter 7. Compose

除了视觉艺术和创意写作，音乐创作是我们认为人类独有的另一种核心创意行为。

要让机器创作出令我们愉悦的音乐，它必须克服前一章中涉及文本时遇到的许多相同的技术挑战。特别是，我们的模型必须能够学习并重新创建音乐的顺序结构，还必须能够从后续音符的离散可能性中进行选择。

然而，音乐生成提出了文本生成所不需要的额外挑战，即音高和节奏。音乐通常是复音的——也就是说，有几个音符流同时在不同的乐器上演奏，它们结合起来创造出不和谐(冲突)或和谐(和谐)的和谐。文本生成只需要我们处理单个文本流，而不是音乐中存在的并行和弦流。

此外，文本生成可以一次处理一个单词。我们必须仔细考虑这是否是处理音乐数据的合适方法，因为听音乐的大部分兴趣是在整个乐团不同节奏之间的相互作用。例如，吉他手可能会演奏一连串更快的音符，而钢琴家则演奏较长的持续和弦。因此，按音符生成音乐是复杂的，因为我们通常不希望所有的乐器同时变换音符。

本章将从简化问题开始，专注于单声道音乐的音乐生成。我们将看到，前一章中关于文本生成的许多RNN技术也可以用于音乐生成，因为这两个任务有许多共同的主题。本章还将介绍注意力机制，它允许我们构建rnn，能够选择关注之前的音符，从而预测接下来会出现哪些音符。最后，我们将解决复调音乐生成的任务，并探索如何部署基于GANs的架构来为多个声音创建音乐。

### Preliminaries

任何处理音乐生成任务的人都必须首先对音乐理论有一个基本的了解。在本节中，我们将介绍阅读音乐所需的基本符号以及如何将其表示为数字，以便将音乐转换为训练生成模型所需的输入数据。

我们将学习notebook 07_01_notation_compose。本书存储库中的Ipynb。另一个入门使用Python生成音乐的优秀资源是Sigurður Skúli的博客文章和附带的GitHub存储库。

我们将使用的原始数据集是一组J.S.巴赫的大提琴组曲的MIDI文件。你可以使用任何你想使用的数据集，但如果你想使用这个数据集，你可以在notebook中找到下载MIDI文件的说明。

要查看和收听模型生成的音乐，您需要一些可以生成乐谱的软件。MuseScore是一个很好的工具，可以免费下载。

###  Musical Notation

我们将使用Python库music21将MIDI文件加载到Python中进行处理。

```python
from music21 import converter
dataset_name = 'cello'
filename = 'cs1-2all'
file = "./data/{}/{}.mid".format(dataset_name, filename)

original_score = converter.parse(file).chordify()
```

![image-20230425163224780](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425163224780.png)

我们使用chordify方法将所有同时演奏的音符压缩为一个声部中的和弦，而不是将它们拆分为多个声部。由于这首曲子是由一种乐器(大提琴)演奏的，所以我们这样做是合理的，尽管有时我们可能希望将这些部分分开来产生本质上是复调的音乐。这带来了更多的挑战，我们将在本章后面讨论。

代码逻辑：循环遍历乐谱，并将乐曲中每个音符(和休止符)的音高和时值提取到两个列表中。和弦中的单个音符被一个点分隔开，这样整个和弦就可以被存储为一个单独的弦。每个音符名称后面的数字表示音符所在的八度，因为音符名称(A到G)重复，所以需要这个八度来唯一标识音符的音高。例如，G2是低于G3的一个倍频程。

 

```python
notes = []
durations = []
for element in original_score.flat:
    if isinstance(element, chord.Chord):
        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
        durations.append(element.duration.quarterLength)
    if isinstance(element, note.Note):
        if element.isRest:
            notes.append(str(element.name))
            durations.append(element.duration.quarterLength)
        else:
            notes.append(str(element.nameWithOctave))
            durations.append(element.duration.quarterLength)
```

这个过程的输出如表7-1所示。

结果数据集现在看起来更像我们之前处理过的文本数据。单词就是音高，我们应该尝试建立一个模型，在给定之前的音高序列的情况下，预测下一个音高。同样的想法也可以应用于持续时间列表。Keras使我们能够灵活地构建一个可以同时处理音高和持续时间预测的模型。

 

![image-20230425164913502](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425164913502.png)

### Your First Music-Generating RNN

为了创建用于训练模型的数据集，我们首先需要给每个基音和持续时间一个整数值(图7-2)，就像我们之前对文本语料库中的每个单词所做的那样。这些值是什么并不重要，因为我们将使用嵌入层将整数查找值转换为向量。

 

![image-20230425165851992](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425165851992.png)

然后，我们通过将数据分割为32个音符的小块，并具有序列中下一个音符的响应变量(独热编码)，用于音高和持续时间，来创建训练集。

 

![image-20230425170106412](C:\Users\cheris\AppData\Roaming\Typora\typora-user-images\image-20230425170106412.png)

我们将要构建的模型是一个带有注意力机制的堆叠LSTM网络。在上一章中，我们看到了如何通过将前一层的隐藏状态作为输入传递到下一层LSTM层来堆叠LSTM层。以这种方式堆叠层可以让模型自由地从数据中学习更复杂的特征。在本节中，我们将介绍注意力机制，它现在是最复杂的序列生成模型的组成部分。它最终产生了transformer，一种完全基于注意力的模型，甚至不需要递归或卷积层。第9章会详细介绍transformer的架构。

现在，让我们专注于将注意力合并到堆叠的LSTM网络中，以尝试在给定之前的音符序列的情况下预测下一个音符。

#### Attention

注意力机制最初应用于文本翻译问题，特别是将英语句子翻译成法语。

在上一章中，我们看到了编码器-解码器网络如何解决这种问题，首先将输入序列通过编码器来生成上下文向量，然后将这个向量通过解码器网络来输出翻译后的文本。这种方法的一个问题是上下文向量可能成为瓶颈。来自源句子开头的信息在到达上下文向量时可能会被稀释，特别是对于长句子。因此，这种类型的编码器-解码器网络有时很难保留所有所需的信息，以使解码器准确地翻译源。

例如，假设我们希望模型将以下句子翻译成德语:I score a penalty in the football match against England。

显然，如果将单词scores替换为missed，整个句子的意思就会发生变化。然而，编码器的最终隐藏状态可能无法充分保留这些信息，因为得分的单词出现在句子的早期。

这个句子的正确翻译是:Ich habe im Fußballspiel gegen England einen Elfmeter erzielt。

如果我们看一下正确的德语翻译，我们可以看到得分(erzielt)这个词实际上出现在句子的末尾!因此，这个模型不仅要保留这样一个事实，即在编码器中判罚得分而不是失分，而且还要一直保留在解码器中。

同样的原理也适用于音乐。要了解某一特定的音乐段落可能会出现哪些音符或音符序列，使用序列中较早的信息可能是至关重要的，而不仅仅是最新的信息。以巴赫第一大提琴组曲的前奏曲为例(图7-4)。

 

![image-20230425172436776](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230425172436776.png)

你认为下一个音符是什么?即使你没有受过音乐训练，你也可以猜出来。如果你说的是G(与曲子的第一个音符相同)，那么你是正确的。你怎么知道的?你可能已经看到每一小节和半小节都以相同的音符开始，并利用这些信息来指导你的决定。我们希望我们的模型能够执行同样的技巧——特别是，我们希望它不仅关心网络现在的隐藏状态，而且还要特别注意网络在8个音符前的隐藏状态，当前一个低G被记录时。

为了解决这一问题，提出了注意机制。而不是只使用编码器RNN的最终隐藏状态作为上下文向量，注意机制允许模型创建上下文向量作为编码器RNN在每个前一个时间步的隐藏状态的加权和。注意机制是将之前的编码器隐藏状态和当前的解码器隐藏状态转换为生成上下文向量的加权总和的一组层。

如果这听起来令人困惑，不要担心!首先，我们将看到如何在一个简单的循环层之后应用注意力机制(即，解决预测巴赫大提琴组曲第一的下一个音符的问题)，然后我们将看到它如何扩展到编码器-解码器网络，我们想要预测后续音符的整个序列，而不仅仅是一个。

 

#### Building an Attention Mechanism in Keras

首先，让我们提醒自己如何使用标准循环层来预测给定前一个音符序列的下一个音符。图7-5显示了输入序列$$(x_1，…，x_n)$$如何一步一步地馈送到层，不断更新层的隐藏状态。输入序列可以是注释嵌入，也可以是来自前一个循环层的隐藏状态序列。循环层的输出是最终隐藏状态，一个与单元数相同长度的向量。然后可以将其馈送到具有softmax输出的Dense层，以预测序列中下一个音符的分布。

 ![image-20230426091711036](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426091711036.png)

图7-6显示了相同的网络，但这一次将注意力机制应用于循环层的隐藏状态。

![image-20230426091851247](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426091851247.png)

```python
notes_in = Input(shape = (None,)) # 音符
durations_in = Input(shape = (None,)) # 持续时间
x1 = Embedding(n_notes, embed_size)(notes_in)  
x2 = Embedding(n_durations, embed_size)(durations_in)
x = Concatenate()([x1,x2]) # 拼接
x = LSTM(rnn_units, return_sequences=True)(x) 
x = LSTM(rnn_units, return_sequences=True)(x)
e = Dense(1, activation='tanh')(x) 
e = Reshape([-1])(e)
alpha = Activation('softmax')(e) 
c = Permute([2, 1])(RepeatVector(rnn_units)(alpha)) 
c = Multiply()([x, c])
c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)
notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c) 
durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
model = Model([notes_in, durations_in], [notes_out, durations_out]) 
att_model = Model([notes_in, durations_in], alpha) 
opti = RMSprop(lr = 0.001)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
              optimizer=opti)
```

![image-20230426101938940](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426101938940.png)

#### Analysis of the RNN with Attention

我们将从从头开始生成一些音乐开始，通过仅用&lt; start &gt;令牌(即，我们告诉模型假设它是从片段的开头开始的)。然后我们可以使用我们在第6章中用于生成文本序列的相同迭代技术来生成一个音乐段落，如下所示:

+ 给定当前序列(音符名称和音符持续时间)，该模型预测下一个音符名称和持续时间的两个分布。
+ 我们从这两个分布中采样，使用温度参数来控制采样过程中的变化程度。
+ 所选音符被存储，其名称和持续时间被附加到相应的序列中。
+ 如果序列的长度现在大于模型被训练的序列长度，我们从序列的开始处删除一个元素。
+ 这个过程用新的序列重复，依此类推，我们希望生成多少音符就生成多少音符。

 

图7-8显示了该模型在训练过程的各个阶段从头生成的音乐示例。我们在本节中的大部分分析将集中在音调预测上，而不是节奏上，因为巴赫的大提琴组曲的和声复杂性更难捕捉，因此更值得研究。但是，您也可以将相同的分析应用于模型的节奏预测，这可能与您可以用于训练该模型的其他风格的音乐(例如鼓音轨)特别相关。

关于图7-8中生成的通道，有几点需要注意。首先，看看随着训练的进行，音乐是如何变得越来越复杂的。首先，该模型通过坚持使用同一组音符和节奏来确保安全。到第10阶段，这个模型已经开始产生小的音符，到第20阶段，它开始产生有趣的节奏，并牢固地建立在一个固定的键(降e大调)上。

 ![image-20230426103252726](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426103252726.png)

每个时间步的预测分布作为热图。图7-8中epoch 20为例，热图如图7-9所示。

![image-20230426103338445](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426103338445.png)

这里值得注意的有趣的一点是，模型已经清楚地了解了哪些音符属于特定的键，因为在不属于该键的音符的分布中存在空白。例如，有一个灰色的间隙沿行注释54(对应于Gb/f#)。这个音在降e大调的乐曲中是极不可能出现的。在生成过程的早期(图的左边)，关键还没有牢固地建立起来，因此在如何选择下一个音符方面有更多的不确定性。随着作品的发展，模型固定在一个键上，某些音符几乎肯定不会出现。

值得注意的是，这个模型并没有在一开始就明确地决定将音乐设置在某个键上，而是在它的过程中不断地创造它，试图选择最适合它之前选择的音符。值得指出的是，该模型已经学习了巴赫的典型风格，即在大提琴上下降到一个低音来结束一个乐句，然后再弹回来开始下一个乐句。看看在音符20左右，乐句以低e调结束，这在巴赫大提琴组曲中很常见，然后在下一个乐句开始时回到更高、更铿锵的乐器音域，这正是模型所预测的。在低E- flat(音高39)和下一个音符之间有一个很大的灰色间隙，这个音符预计在音高50左右，而不是继续在乐器的深处隆隆作响。

最后，我们应该检查我们的注意力机制是否像预期的那样工作。图7-10所示为网络在生成序列中各点计算出的alpha向量元素值。横轴表示生成的音符序列;纵轴显示了当沿水平轴(即alpha向量)预测每个音符时，网络的注意力集中在哪里。正方形越暗，对序列中与此点对应的隐藏状态的关注就越大。

 

![image-20230426103628561](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426103628561.png)

我们可以看到，对于作品的第二个音符(B-3 =降b)，网络选择将几乎所有的注意力放在作品的第一个音符也是B-3的事实上。这是有道理的;如果你知道第一个音符是降B，你可能会用这个信息来决定下一个音符。

当我们在接下来的几个音符中移动时，神经网络将它的注意力大致均匀地分散在之前的音符上——然而，它很少把注意力放在超过六个音符之前的音符上。再说一次，这是有道理的;在前六个隐藏状态中可能包含足够的信息来理解这个短语应该如何继续。也有网络选择忽略附近某个音符的例子，因为它不会在理解短语时添加任何额外的信息。

例如，看一下图表中心的白色方框，注意中间有一条方框，它切断了回顾前面四到六个音符的通常模式。为什么网络在决定如何继续这个短语时愿意选择忽略这个注释呢?如果你看一下它对应的是哪个音符，你会发现它是三个E-3音符中的第一个。模型选择忽略这一点，因为在此之前的音符也是降e，低一个八度(E-2)。此时网络的隐藏状态将为模型提供足够的信息来理解降e是这段话中的一个重要音符，因此模型不需要注意后续的更高的降e，因为它没有添加任何额外的信息。

更多的证据表明，该模型已经开始理解八度的概念，可以在下方和右侧的绿色框中看到。在这里，模型选择忽略低G (G2)，因为在此之前的音符也是G (G3)，高一个八度。记住，我们并没有告诉模型，哪些音符是通过八度相关联的——它只是通过研究巴赫的音乐，自己解决了这个问题，这很了不起。

####  Attention in Encoder–Decoder Networks

注意机制是一个强大的工具，可以帮助网络决定循环层的哪些先前状态对于预测序列的延续是重要的。到目前为止，我们已经看到了提前一个音符的预测。然而，我们也可能希望将注意力构建到编码器-解码器网络中，在那里我们通过使用RNN解码器来预测未来音符的序列，而不是一次构建一个音符序列。

回顾一下，图7-11显示了一个标准的编码器-解码器模型在没有注意的情况下的样子——我们在第6章中介绍的那种。图7-12显示了相同的网络，但在编码器和解码器之间增加了注意机制。

 ![image-20230426151200476](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426151200476.png)

![image-20230426151244452](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426151244452.png)

注意机制的工作方式与我们之前看到的完全相同，只是有一点改变:解码器的隐藏状态也被纳入到机制中，这样模型不仅可以通过之前的编码器隐藏状态，还可以从当前的解码器隐藏状态决定将注意力集中在哪里。图7-13显示了编码器-解码器框架中注意模块的内部工作原理。

 ![image-20230426151449798](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426151449798.png)

虽然在编码器-解码器网络中存在许多注意机制的副本，但它们都共享相同的权值，因此在需要学习的参数数量上没有额外的开销。唯一的变化是，现在，解码器隐藏状态被滚动到注意力计算中(图中的红线)。这稍微改变了方程，加入了一个额外的索引(i)来指定解码器的步长。

还要注意在图7-11中我们如何使用编码器的最终状态来初始化解码器的隐藏状态。在有注意的编码器-解码器中，我们使用循环层的内置标准初始化器来初始化解码器。上下文向量c与传入数据y连接起来，形成一个扩展的数据向量，进入解码器的每个单元。因此，我们将上下文向量视为输入解码器的附加数据。

 

### Generating Polyphonic Music

我们在本节中探索的带有注意机制框架的RNN在单线(单音)音乐中效果很好，但它能适应多线(复音)音乐吗?

RNN框架当然足够灵活，可以通过循环机制同时生成多条音乐线。但就目前而言，我们目前的数据集并没有很好地为此设置，因为我们将和弦存储为单个实体，而不是由多个单独的音符组成的部分。例如，我们当前的RNN无法知道C大调和弦(C、E和G)实际上与a小调和弦(a、C和E)非常接近——只有一个音符需要改变，即G变为a。

相反，它将两者视为两个不同的元素，可以独立预测。理想情况下，我们希望设计一个网络，可以接受多个渠道的音乐作为单独的流，并学习这些流应该如何相互作用，以产生好听的音乐，而不是不和谐的噪音。

这听起来是不是有点像生成图像?对于图像生成，我们有三个通道(红、绿、蓝)，我们希望网络学习如何组合这些通道来生成漂亮的图像，而不是随机的像素化噪声。事实上，正如我们将在下一节看到的，我们可以将音乐生成直接视为图像生成问题。这意味着我们可以将同样的基于卷积的技术应用于音乐图像生成问题，而不是使用循环网络，特别是gan。在我们探索这个新建筑之前，我们只有足够的时间去参观音乐厅，那里的演出即将开始。

 

### The Musical Organ

指挥家在指挥台上敲了两下指挥棒。演出就要开始了。在他面前坐着一支管弦乐队。然而，这支乐团并不打算演奏贝多芬的交响乐或柴可夫斯基的序曲。这个管弦乐队在演出期间现场创作原创音乐，完全由一组演奏者向舞台中央的一个巨大的管风琴(简称MuseGAN)发出指令，它将这些指令转化为美妙的音乐，为观众带来愉悦。管弦乐队可以通过训练来演奏特定风格的音乐，而且没有两次演出是相同的。

管弦乐队的128名演奏者被分成4个平均的小组，每组32名演奏者。每个部分都向MuseGAN提供指示，并在管弦乐队中负有明确的责任。

风格组负责制作演出的整体音乐风格。在许多方面，它是所有部分中最简单的工作，因为每个演奏者只需要在音乐会开始时生成一个指令，然后在整个演出过程中不断地向博物馆提供信息。凹槽部分有类似的工作，但每个播放器产生几个指令:一个为每个不同的音乐轨道，由MuseGAN输出。

例如，在一场音乐会中，每个groove部分的成员制作了五个指令，分别对应人声、钢琴、弦乐、贝斯和鼓的音轨。因此，他们的工作是为每一个独立的器乐声音提供槽，然后在整个演出中保持不变。风格和凹槽部分在整个作品中没有改变它们的指示。表演的动态元素是由最后两个部分提供的，这确保了音乐随着每个小节的变化而不断变化。小节(或小节)是一个小的音乐单位，包含固定的、少量的节拍。例如，如果你能跟着一段音乐数1、2、1、2，那么每小节就有两拍，你可能在听进行曲。如果你能数1、2、3、1、2、3，那么每小节有三拍，你可能正在听华尔兹。

和弦部分的演奏者在每小节开始时改变他们的指示。这样做的效果是给每个小节一个独特的音乐特征，例如，通过一个和弦的变化。和弦部分的演奏者每小节只产生一个指令，然后应用于每个器乐轨道。旋律部分的乐手是最累人的工作，因为他们在整首曲子的每个小节开始时对每个器乐音轨给出不同的指示。这些玩家对音乐有最精细的控制，因此这可以被认为是提供旋律兴趣的部分。这就完成了对管弦乐队的描述。

我们可以将各部分的职责总结如表7-2所示。

 ![image-20230426154336388](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426154336388.png)

根据当前的128条指令(每个播放器一条指令)，由MuseGAN生成下一个音乐小节。训练MuseGAN做到这一点并不容易。最初，乐器只会产生可怕的噪音，因为它无法理解如何解释指令来产生与真正的音乐没有区别的小节。

这就是指挥员的作用。当音乐与真实音乐明显不同时，指挥会告诉MuseGAN，然后MuseGAN调整其内部线路，以便下次更有可能骗过指挥。指挥和博物馆使用的过程与我们在第4章看到的完全相同，Di和Gene一起工作，不断改进Gene拍摄的动物照片。MuseGAN的演奏者在世界各地巡回演出，在有足够的现有音乐来训练MuseGAN的地方举办任何风格的音乐会。在下一节中，我们将看到如何使用Keras构建MuseGAN，以学习如何生成逼真的复调音乐。

 

## Chapter 8. Play

2018年3月，David Ha和Jürgen Schmidhuber发表了他们的“世界模型”论文。这篇论文展示了如何训练一个模型，该模型可以通过在自身生成的幻觉梦境中进行实验，而不是在环境本身中学习如何执行特定任务。当与强化学习等其他机器学习技术一起应用时，这是一个很好的例子，说明了生成式建模如何用于解决实际问题。

该架构的一个关键组件是生成模型，在给定当前状态和动作的情况下，该模型可以构建下一个可能状态的概率分布。在通过随机运动建立了对环境基本物理的理解之后，该模型能够完全在其自身对环境的内部表示中从头开始训练自己的新任务。这种方法在两项测试中都获得了世界上最好的分数。

在本章中，我们将详细探索该模型，并展示如何创建这种惊人的尖端技术的自己版本。基于原始论文，我们将构建一个强化学习算法，学习如何以尽可能快的速度在赛道上驾驶汽车。虽然我们将使用2D计算机模拟作为我们的环境，但同样的技术也可以应用于现实世界的场景，在真实环境中测试策略是昂贵的或不可实现的。

然而，在我们开始构建模型之前，我们需要仔细了解一下强化学习的概念和OpenAI Gym平台。

 

### Reinforcement Learning

强化学习可以定义如下:

+ 强化学习(RL)是机器学习的一个领域，旨在训练智能体在给定环境中针对特定目标进行最佳表现。

判别建模和生成建模的目标都是最小化观测数据集上的损失函数，而强化学习的目标是最大化智能体在给定环境中的长期奖励。它通常被描述为机器学习的三大分支之一，与监督学习(使用标记数据进行预测)和无监督学习(从无标记数据中学习结构)并列。

让我们首先介绍一些与强化学习相关的关键术语:

+ Environment：智能体运行的世界。它定义了一组规则，根据智能体的先前动作和当前游戏状态，控制游戏状态更新过程和奖励分配。例如，如果我们正在教强化学习算法下国际象棋，环境将由控制给定行动(例如，移动e4)如何影响下一个游戏状态(棋盘上棋子的新位置)的规则组成，还将指定如何评估给定位置是否被将死，并在获胜棋手获胜后分配1的奖励。

+ Agent：在环境中执行操作的智能体。
+ Game state：代表代理可能的特定情况的数据遭遇(也称为状态)，例如，一个特定的棋盘配置附带的游戏信息，如哪个玩家会采取下一步行动。
+ Action：一个智能体可以采取的可行行动。
+ Reward：奖励
+ Episode：在环境中运行一次智能体;这也被称为rollout。
+ Timestep：对于离散事件环境，所有状态、动作和奖励都被标下标，以显示它们在时间步t的值。


这些定义之间的关系如图8-1所示。

 ![image-20230426165238864](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426165238864.png)

首先用当前游戏状态s初始化环境。在时间步t，智能体接收当前游戏状态s，并使用它来决定下一个最佳动作a，然后执行该动作。给定这个动作，环境然后计算下一个状态s并奖励r，并将这些传递回智能体，以便再次开始循环。循环继续，直到满足事件的结束标准(例如，经过给定数量的时间步长或代理赢/输)。

我们如何设计一个智能体来最大化给定环境中的奖励总和?我们可以构建一个智能体，它包含一套如何响应任何给定游戏状态的规则。然而，随着环境变得更加复杂，这很快就变得不可行，并且永远不允许我们在特定任务中构建具有超人能力的智能体，因为我们正在硬编码规则。强化学习包括创建一个智能体，该智能体可以通过反复游戏在复杂环境中学习最优策略——本章将使用它来构建我们的智能体。

现在我将介绍OpenAI Gym，赛车环境之家，我们将使用它来模拟汽车在赛道上行驶。

 

### OpenAI Gym

OpenAI Gym是一个用于开发强化学习算法的工具包，可以作为Python库使用。

该库中包含几个经典的强化学习环境，如CartPole和Pong，以及提出更复杂挑战的环境，如训练智能体在不平坦的地形上行走或赢得Atari游戏。所有的环境都提供了step方法，你可以通过它提交给定的操作;环境将返回下一个状态和奖励。通过使用智能体选择的操作反复调用step方法，您可以在环境中播放一个片段。

除了每个环境的抽象机制外，OpenAI Gym还提供了允许您观看智能体在给定环境中执行的图形。这对于调试和查找智能体可以改进的地方很有用。我们将利用OpenAI Gym中的赛车环境。让我们看看如何为这个环境定义游戏状态、动作、奖励和情节:

+ Game state
  + A 64 × 64–pixel RGB image depicting an overhead view of the track and car.
+ Action
  + A set of three values: the steering direction (–1 to 1), acceleration (0 to 1), and braking (0 to 1). The agent must set all three values at each timestep.
+ Reward
  + A negative penalty of –0.1 for each timestep taken and a positive reward of 1000/N if a new track tile is visited, where N is the total number of tiles that make up the track.
+ Episode
  + The episode ends when either the car completes the track, drives off the edge of the environment, or 3,000 timesteps have elapsed.

图8-2以图形形式展示了这些概念。请注意，汽车从它的视角看不到轨道，但我们应该想象一个漂浮在轨道上方的智能体从鸟瞰图控制汽车。

 

![image-20230426171429540](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426171429540.png)

### World Model Architecture

在探索构建每个组件所需的详细步骤之前，我们现在将介绍我们将用于构建通过强化学习进行学习的代理的整个架构的高级概述。该解决方案由三个不同的部分组成，分别进行训练，如图8-3所示。

+ V
  + A variational autoencoder.
+ M
  + A recurrent neural network with a mixture density network (MDN-RNN).
+ C
  + A controller.

![image-20230426172302701](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230426172302701.png)

#### The Variational Autoencoder

当你在开车时做决定时，你不会主动分析视图中的每个像素，而是将视觉信息压缩为少量的潜在实体，例如道路的直线度、即将到来的转弯以及你相对于道路的位置，以通知你的下一步行动。

 我们在第3章中看到，VAE如何通过最小化重构误差和KL散度，将一个高维输入图像压缩为一个近似服从标准多元正态分布的潜在随机变量。这确保了潜空间是连续的，并且我们能够轻松地从中采样以生成有意义的新观测。

在赛车示例中，VAE将64 × 64 × 3 (RGB)输入图像压缩为32维正态分布随机变量，参数为mu和log_var。这里，log_var是分布方差的对数。我们可以从这个分布中采样，以产生表示当前状态的潜在向量z。这被传递到网络的下一部分，MDN-RNN。

####  The MDN-RNN

当你开车时，随后的每一个观察对你来说都不是一个完全的惊喜。如果当前的观察显示前方道路左转，而你把方向盘转到左边，你期望下一次观察显示你仍然与道路保持一致。

如果你没有这种能力，你的驾驶可能会在整个道路上蜿蜒，因为你无法看到稍微偏离中心会在下一个时间步变得更糟，除非你现在就做些什么。

这种前瞻性思维是MDN-RNN的工作，这是一个试图根据前一个潜在状态和前一个动作预测下一个潜在状态分布的网络。具体来说，MDN-RNN是一个具有256个隐藏单元的LSTM层，后面是一个混合密度网络(MDN)输出层，它允许下一个潜在状态实际上可以从几个正态分布中的任何一个中提取。“世界模型”论文的作者之一David Ha将同样的技术应用于手写生成任务，如图8-4所示，以描述下一个笔尖可能落在任何一个明显的红色区域的事实。

 

![image-20230427110040387](https://gitee.com/chjjj666/mkdown-images/raw/master/imgs/image-20230427110040387.png)

在赛车的例子中，我们允许从五个正态分布中的任意一个提取下一个观察到的潜在状态的每个元素。

#### The Controller

到目前为止，我们还没有提到任何关于选择动作的内容。这个责任在于控制者。

控制器是一个密集连接的神经网络，其中输入是z(从VAE编码的分布中采样的当前潜状态)和RNN的隐藏状态的连接。三个输出神经元对应于三个动作(转弯、加速、刹车)，并被缩放到适当的范围内下降。我们需要使用强化学习来训练控制器，因为没有训练数据集可以告诉我们某个动作是好的，另一个动作是坏的。

相反，智能体将需要通过反复实验自己发现这一点。正如我们将在本章后面看到的，“世界模型”论文的关键是它演示了如何在智能体自己的环境生成模型中进行这种强化学习，而不是在OpenAI Gym环境中。换句话说，它发生在智能体对环境行为的幻觉版本中，而不是真实的东西。

为了理解这三个组件的不同角色以及它们如何一起工作，我们可以想象它们之间的对话:

+ VAE(查看最新的64 × 64 × 3观察):这看起来像一条笔直的道路，有一个轻微的左转弯接近，汽车面向道路的方向(z)。

+ RNN:根据描述(z)和控制器在最后一个时间步(动作)选择加速的事实，我将更新我的隐藏状态，以便预测下一个观察仍然是直线道路，但在视图中有更多的左转。

+ 控制器:基于VAE (z)的描述和RNN (h)的当前隐藏状态，我的神经网络输出[0.34,0.8,0]作为下一个动作。

然后，控制器的动作被传递给环境，环境返回一个更新后的观测值，然后循环再次开始。

有关该模型的更多信息，也有一个优秀的在线交互式解释。

####  Setup
