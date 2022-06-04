ffmpeg -framerate 1 -pattern_type glob -i './images/GAN/*.png' -c:v libx264 -pix_fmt yuv420p ./videos/GAN.mp4;
ffmpeg -framerate 1 -pattern_type glob -i './images/WGAN/*.png' -c:v libx264 -pix_fmt yuv420p ./videos/WGAN.mp4;
ffmpeg -framerate 1 -pattern_type glob -i './images/WGANGP/*.png' -c:v libx264 -pix_fmt yuv420p ./videos/WGANGP.mp4;
