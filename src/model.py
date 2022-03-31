#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import torch
from os.path import isfile
from torchsummary import summary


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        input_height=project_parameters.input_height,
        in_chans=project_parameters.in_chans,
        generator_feature_dim=project_parameters.generator_feature_dim,
        latent_dim=project_parameters.latent_dim,
        discriminator_feature_dim=project_parameters.discriminator_feature_dim)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class Encoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim,
                 add_final_conv) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_chans,
                      out_channels=out_chans,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        input_height, out_chans = input_height / 2, out_chans
        while input_height > 4:
            in_channels = out_chans
            out_channels = out_chans * 2
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            out_chans *= 2
            input_height /= 2
        if add_final_conv:
            layers.append(
                nn.Conv2d(in_channels=out_chans,
                          out_channels=latent_dim,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        out_chans, target_size = out_chans // 2, 4
        while target_size != input_height:
            out_chans *= 2
            target_size *= 2
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=out_chans,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_chans))
        layers.append(nn.ReLU(inplace=True))
        target_size = 4
        while target_size < input_height // 2:
            layers.append(
                nn.ConvTranspose2d(in_channels=out_chans,
                                   out_channels=out_chans // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_chans // 2))
            layers.append(nn.ReLU(inplace=True))
            out_chans //= 2
            target_size *= 2
        layers.append(
            nn.ConvTranspose2d(in_channels=out_chans,
                               out_channels=in_chans,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, input_height, in_chans, generator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        self.encoder = Encoder(input_height=input_height,
                               in_chans=in_chans,
                               out_chans=generator_feature_dim,
                               latent_dim=latent_dim,
                               add_final_conv=True)
        self.decoder = Decoder(input_height=input_height,
                               in_chans=in_chans,
                               out_chans=generator_feature_dim,
                               latent_dim=latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat


class Discriminator(nn.Module):
    def __init__(self, input_height, in_chans, discriminator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        layers = Encoder(input_height=input_height,
                         in_chans=in_chans,
                         out_chans=discriminator_feature_dim,
                         latent_dim=latent_dim,
                         add_final_conv=True)
        layers = list(layers.layers.children())
        self.extractor = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.activation_function = nn.Sigmoid()

    def forward(self, x):
        features = self.extractor(x)
        y = self.activation_function(self.classifier(features))
        return features, y


class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config,
                 input_height, in_chans, generator_feature_dim, latent_dim,
                 discriminator_feature_dim) -> None:
        super().__init__(optimizers_config, lr, lr_schedulers_config)
        self.generator_xy = Generator(
            input_height=input_height,
            in_chans=in_chans,
            generator_feature_dim=generator_feature_dim,
            latent_dim=latent_dim)
        self.generator_yx = Generator(
            input_height=input_height,
            in_chans=in_chans,
            generator_feature_dim=generator_feature_dim,
            latent_dim=latent_dim)
        self.discriminator_x = Discriminator(
            input_height=input_height,
            in_chans=in_chans,
            discriminator_feature_dim=discriminator_feature_dim,
            latent_dim=latent_dim)
        self.discriminator_y = Discriminator(
            input_height=input_height,
            in_chans=in_chans,
            discriminator_feature_dim=discriminator_feature_dim,
            latent_dim=latent_dim)
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizers_g_xy = self.parse_optimizers(
            params=self.generator_xy.parameters())
        optimizers_g_yx = self.parse_optimizers(
            params=self.generator_yx.parameters())
        optimizers_d_x = self.parse_optimizers(
            params=self.discriminator_x.parameters())
        optimizers_d_y = self.parse_optimizers(
            params=self.discriminator_y.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers_g_xy = self.parse_lr_schedulers(
                optimizers=optimizers_g_xy)
            lr_schedulers_g_yx = self.parse_lr_schedulers(
                optimizers=optimizers_g_yx)
            lr_schedulers_d_x = self.parse_lr_schedulers(
                optimizers=optimizers_d_x)
            lr_schedulers_d_y = self.parse_lr_schedulers(
                optimizers=optimizers_d_y)
            return [
                optimizers_g_xy[0], optimizers_g_yx[0], optimizers_d_x[0],
                optimizers_d_y[0]
            ], [
                lr_schedulers_g_xy[0], lr_schedulers_g_yx[0],
                lr_schedulers_d_x[0], lr_schedulers_d_y[0]
            ]
        else:
            return [
                optimizers_g_xy[0], optimizers_g_yx[0], optimizers_d_x[0],
                optimizers_d_y[0]
            ]

    def weights_init(self, module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, x):
        y_hat = self.generator_xy(x)
        x_hat = self.generator_yx(y_hat)
        return y_hat, x_hat

    def shared_step(self, batch):
        x, y = batch
        # x -> x_y -> x_y_x
        x_y = self.generator_xy(x)
        x_y_x = self.generator_yx(x_y)
        # y _> y_x -> y_x_y
        y_x = self.generator_yx(y)
        y_x_y = self.generator_xy(y_x)
        return x, y, x_y, x_y_x, y_x, y_x_y

    def calculate_generator_loss(self, x, y, x_y, x_y_x, y_x, y_x_y):
        #generator_xy
        feat_y, proba_y = self.discriminator_y(y)
        feat_y_hat, proba_y_hat = self.discriminator_y(x_y)
        g_loss_feat_y = self.l1_loss(feat_y_hat, feat_y)
        g_loss_proba_y = self.bce_loss(proba_y_hat,
                                       torch.ones_like(input=proba_y_hat))
        g_loss_identity_xy = self.l1_loss(self.generator_xy(y), y)
        g_loss_consistency_x_y_x = self.l1_loss(x_y_x, x)

        #generator_yx
        feat_x, proba_x = self.discriminator_x(x)
        feat_x_hat, proba_x_hat = self.discriminator_x(y_x)
        g_loss_feat_x = self.l1_loss(feat_x_hat, feat_x)
        g_loss_proba_x = self.bce_loss(proba_x_hat,
                                       torch.ones_like(input=proba_x_hat))
        g_loss_identity_yx = self.l1_loss(self.generator_yx(x), x)
        g_loss_consistency_y_x_y = self.l1_loss(y_x_y, y)

        #generator loss
        g_loss = g_loss_feat_y + \
                    g_loss_proba_y + \
                    g_loss_identity_xy + \
                    g_loss_consistency_x_y_x + \
                    g_loss_feat_x + \
                    g_loss_proba_x + \
                    g_loss_identity_yx + \
                    g_loss_consistency_y_x_y
        return g_loss

    def calculate_discriminator_loss(self, x, y, x_y, x_y_x, y_x, y_x_y):
        #discriminator y
        feat_y, proba_y = self.discriminator_y(y)
        feat_y_hat, proba_y_hat = self.discriminator_y(x_y.detach())
        d_loss_y_real = self.bce_loss(proba_y, torch.ones_like(input=proba_y))
        d_loss_y_fake = self.bce_loss(proba_y_hat,
                                      torch.zeros_like(input=proba_y_hat))

        #discriminator x
        feat_x, proba_x = self.discriminator_x(x)
        feat_x_hat, proba_x_hat = self.discriminator_x(y_x.detach())
        d_loss_x_real = self.bce_loss(proba_x, torch.ones_like(input=proba_x))
        d_loss_x_fake = self.bce_loss(proba_x_hat,
                                      torch.zeros_like(input=proba_x_hat))

        #discriminator loss
        d_loss = d_loss_y_real + \
                    d_loss_y_fake + \
                    d_loss_x_real + \
                    d_loss_x_fake
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)

        if optimizer_idx == 0 or optimizer_idx == 1:
            g_loss = self.calculate_generator_loss(x=x,
                                                   y=y,
                                                   x_y=x_y,
                                                   x_y_x=x_y_x,
                                                   y_x=y_x,
                                                   y_x_y=y_x_y)

            self.log('train_loss',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log('train_loss_generator',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return g_loss

        if optimizer_idx == 2 or optimizer_idx == 3:
            d_loss = self.calculate_discriminator_loss(x=x,
                                                       y=y,
                                                       x_y=x_y,
                                                       x_y_x=x_y_x,
                                                       y_x=y_x,
                                                       y_x_y=y_x_y)

            self.log('train_loss_discriminator',
                     d_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)

        #generator loss
        g_loss = self.calculate_generator_loss(x=x,
                                               y=y,
                                               x_y=x_y,
                                               x_y_x=x_y_x,
                                               y_x=y_x,
                                               y_x_y=y_x_y)

        #discriminator loss
        d_loss = self.calculate_discriminator_loss(x=x,
                                                   y=y,
                                                   x_y=x_y,
                                                   x_y_x=x_y_x,
                                                   y_x=y_x,
                                                   y_x_y=y_x_y)

        #reinitailize discriminator
        if d_loss.item() < 1e-5:
            self.discriminator_x.apply(self.weights_init)
            self.discriminator_y.apply(self.weights_init)

        self.log('val_loss', g_loss)
        self.log('val_loss_generator', g_loss, prog_bar=True)
        self.log('val_loss_discriminator', d_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, x_y, x_y_x, y_x, y_x_y = self.shared_step(batch=batch)

        #generator loss
        g_loss = self.calculate_generator_loss(x=x,
                                               y=y,
                                               x_y=x_y,
                                               x_y_x=x_y_x,
                                               y_x=y_x,
                                               y_x_y=y_x_y)

        self.log('test_loss', g_loss)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.input_height,
                        project_parameters.input_height),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.input_height,
                   project_parameters.input_height)

    # get model output
    loss, x_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(x_hat.shape))
