import matlab.engine
eng = matlab.engine.start_matlab()
eng.eval("disp('Hello from MATLAB!')", nargout=0)
