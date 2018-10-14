The disseration was coded using the Pycharm IDE which handles importing Python files from other folders internally.
To run without IDE, the following snippet codes need to be added to files that import from other folders:

import sys
sys.path.extend([WORKING_DIRECTORY, FILE_DIRECTORY])

where WORKING_DIRECTORY is the root folder of the project, FILE_DIRECTORY is the subfolder where the file to be run is contained.