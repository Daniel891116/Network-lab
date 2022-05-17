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
<table>
  <tr>
    <th>Mode </th>
    <th>Effect</th>
  </tr>
  <tr>
    <td>0(default)</td>
    <td>None</td>
  </tr>
  <tr>
    <td>1</td>
    <td>hand gesture</td>
  </tr>
  <tr>
    <td>2</td>
    <td>object recognition</td>
  </tr>
  <tr>
    <td>3</td>
    <td>pose detection</td>
  </tr>
</table>

## Video LINK
- https://youtu.be/S4Wq9ryygtE