#!/bin/bash
source '/usr/local/python37/bin/virtualenvwrapper.sh'
echo 'start book process'
workon ttenv
echo 'set env'
cd /home/pi/gitproject/LibraryRoomBook/
echo 'start main.py'
python main.py
