# Dota2 - ML

A machine learning project for Dota 2 – a multiplayer online battle arena (MOBA) game 
that can predict the probability of winning a match given the
characters chosen.

The project explores a number of features include: heroes synergistic and antagonistic relationships, players past performance, match duration and pick order of heroes. These features were tested and compared between five
different supervised machine learning algorithms.

A test accuracy of 70% were achieved through incorporating these new features
into the base model. Furthermore, it was found that through limiting the data to matches
under 30 minutes only, a 77% test accuracy was obtained

Addtionally, the project includes an interface built using the library _Tkinter_ that can help players in choosing the
optimal character that will give them the edge in a match.

### Notes

The disseration was coded using the Pycharm IDE which handles importing Python files from other folders internally.
To run without IDE, the following snippet codes need to be added to files that import from other folders:

```
import sys
sys.path.extend([WORKING_DIRECTORY, FILE_DIRECTORY])
```

where WORKING_DIRECTORY is the root folder of the project, FILE_DIRECTORY is the subfolder where the file to be run is contained.
