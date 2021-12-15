If you want to run the Stability experiment with 10 epochs and signal f(x,y)=x*y/ random band-limited signal

1. mkdir ../input_rad
2. mkdir ../input_rad/processed
3. mkdir ../input_rad/raw
4. python DataLoader_rad_grid.py
5. python Run_Epocs.py/RunRandomBandlimited_Epochs.py

To-Dos:
    - make the epochs variable (fixed to 10), no problem
    - make the signal variable (only two signals), no problem
    - make the radius variable (fixed to 0.1,0.5,0.9), maybe problem
