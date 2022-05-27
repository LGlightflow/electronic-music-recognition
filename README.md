# electronic-music-recognition
简简单单的一个电子音乐风格识别，数据及准确度未优化

基于Valerio Velardo的教程

数据集是在网上的音乐歌单中自己筛选的
音乐种类有dnb dubstep house midtempo techno trance trap

用cut.py生成相应相应的音乐片段
在用extract_data.py生成含mfcc的json文件
模型也放了上来

因电子音乐的元素很多且有些歌带了人声而没处理掉（特别是midtempo），故实际应用时不准确
