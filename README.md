# Network-lab

## In Jetson nano
```bash
# Install protobuf compiler
$ sudo apt-get install protobuf-compiler
# Install buildtools
$ sudo apt-get install build-essential make
# Install packages
$ pip3 install -r requirements.txt
```
## How to run

In Jetson nano
- run gRPC server
```bash
$ python3 week12_HW.py
```
In your computer
- run ffplay at 1st terminal
```bash
$ ffplay -fflags nobuffer rtmp://{your server IP}/rtmp/live
```
- send gRPC command at 2rd terminal
```bash
$ python3 client --ip {your server IP} --mode {mode}
```