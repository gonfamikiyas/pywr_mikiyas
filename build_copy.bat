set LIBRARY=C:/Users/bczol/py11/py11
set LIBRARY_INC=%LIBRARY%\include
set LIBRARY_LIB=%LIBRARY%\lib
python setup.py build_ext -I"%LIBRARY_INC%" -L"%LIBRARY_LIB%" --inplace --with-glpk develop --enable-debug develop
