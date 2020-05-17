# 图书馆预约工具
## 项目简介
研习室预定懒人包
## 功能
按配置文件定时自动预定研习室
## 环境需求
- 系统环境：ubuntu18.04
- 编译/脚本环境
  - opencv3.4.3
  - python3.7.2
    - tensorflow
    - numpy
    - requests
    - beautifulsoup4
    - selenium
- chrome/chromium+driver
## 部署步骤
1. 安装好chromium和对应驱动
2. 初始化环境（python，opencv）
3. 下载本项目
4. 修改配置
5. 使用cron定时或手动启动程序
## 目录结构
```shell
├─image_code（验证码图片资源，另行下载）       
│  ├─test
│  ├─test_cut       
│  ├─training       
│  ├─train_cut      
│  ├─train_source   
│  ├─verification_cut   
│  └─verification_source
├─Mypackages
│  └─NNmodule
│    ├─ckpt
│    └─data_set 
├─other
└─项目文档
```
## 版本更新
[更新日志](./项目文档/changelog.md)
## 协议
[MIT协议](./LICENSE.md)