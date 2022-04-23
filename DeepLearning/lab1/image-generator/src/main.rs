use std::{
    fs::File,
    io::{self, Read},
    mem, vec,
};

fn main() -> io::Result<()> {
    let mut labels_file = File::open("../data/MNIST/raw/train-labels-idx1-ubyte")?;
    let mut images_file = File::open("../data/MNIST/raw/train-images-idx3-ubyte")?;

    // 检验标签文件头
    let mut buffer: Vec<u8> = vec![0; 4 + 4];
    let readed_len = labels_file.read(&mut buffer)?;
    assert_eq!(4 + 4, readed_len);
    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    assert_eq!(2049, magic);
    let image_count = u32::from_be_bytes(buffer[4..8].try_into().unwrap());
    println!(
        "[lables file]: magic = {}, image_count = {}",
        magic, image_count
    );

    // 检验图像文件头
    let mut buffer: Vec<u8> = vec![0; 4 + 4 + 4 + 4];
    let readed_len = images_file.read(&mut buffer)?;
    assert_eq!(4 + 4 + 4 + 4, readed_len);
    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    assert_eq!(2051, magic);
    let image_count = u32::from_be_bytes(buffer[4..8].try_into().unwrap());
    let image_height = u32::from_be_bytes(buffer[8..12].try_into().unwrap());
    let image_width = u32::from_be_bytes(buffer[12..16].try_into().unwrap());
    println!(
        "[images file] magic = {}, image_count = {}, image_height = {}, image_width = {}",
        magic, image_count, image_height, image_width
    );

    // 读入图片的数量
    let expected_image_count = 10;
    let mut image_index = 0;

    loop {
        // 读取图片标签
        let mut buffer: Vec<u8> = vec![0; mem::size_of::<u8>()];
        let readed_len = labels_file.read(&mut buffer)?;
        if readed_len == 0 {
            println!("lables read end");
            break;
        }
        let image_label = buffer[0] as u32;

        // 读取图片像素
        let mut buffer: Vec<u8> = vec![0; image_height as usize * image_width as usize];
        let readed_len = images_file.read(&mut buffer)?;
        if readed_len == 0 {
            println!("images read end");
            break;
        }
        (0..buffer.len()).for_each(|i| buffer[i] = u8::MAX - buffer[i]);

        // 生成图片文件名，并将图片写入文件
        let mut image_filename = String::from("../data/MNIST/png/image");
        image_filename.push_str(&image_index.to_string());
        image_filename.push_str(" - ");
        image_filename.push_str(&image_label.to_string());
        image_filename.push_str(".png");
        image::save_buffer(
            image_filename,
            &buffer,
            image_width,
            image_height,
            image::ColorType::L8,
        )
        .unwrap();

        image_index += 1;
        if image_index >= expected_image_count {
            break;
        }
    }

    Ok(())
}
