"""
Component tests will not be initialized as python modules (actually they are initialized as
python modules but the individual classes/testcases are not imported in the __init__.py files)
to not run them with all the other tests, since these are copied from the official tensorflow
models garden GitHub repository and will likely take a lot of time to execute.
When extensive changes have been made to the source files, the integration can make sense again.
"""
