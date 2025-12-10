.. birdnet documentation master file, created by
   sphinx-quickstart on Wed Dec 10 07:43:17 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

birdnet documentation
=====================

Documentation for the birdnet package.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   birdnet
   models

Introduction
------------

A Python library for identifying bird species by their sounds.

The library is geared towards providing a robust workflow for ecological data analysis in bioacoustic projects. While it covers essential functionalities, it doesn’t include all the features found in BirdNET-Analyzer, which is available [here](https://github.com/birdnet-team/BirdNET-Analyzer). Some features might only be available in the BirdNET Analyzer and not in this package.

This project is under active development, so you might encounter changes that could affect your current workflow. We recommend checking for updates regularly.

The package is also available as an R package at: [birdnetR](https://github.com/birdnet-team/birdnetR).

Citing BirdNET-Analyzer
-----------------------

Feel free to use BirdNET for your acoustic analyses and research. If you do, please cite as:

.. code-block:: bibtex

   @article{kahl2021birdnet,
     title={BirdNET: A deep learning solution for avian diversity monitoring},
     author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
     journal={Ecological Informatics},
     volume={61},
     pages={101236},
     year={2021},
     publisher={Elsevier}
   }

About
-----

Developed by the `K. Lisa Yang Center for Conservation Bioacoustics <https://www.birds.cornell.edu/ccb/>`_ at the `Cornell Lab of Ornithology <https://www.birds.cornell.edu/home>`_ in collaboration with `Chemnitz University of Technology <https://www.tu-chemnitz.de/index.html>`_.

Go to https://birdnet.cornell.edu to learn more about the project.

Want to use BirdNET to analyze a large dataset? Don't hesitate to contact us: ccb-birdnet@cornell.edu

We also have a discussion forum on `Reddit <https://www.reddit.com/r/BirdNET_Analyzer/>`_ if you have a general question or just want to chat.

Have a question, remark, or feature request? Please start a new issue thread to let us know. Feel free to submit a pull request.

More tools and resources
------------------------

We also provide Python and R packages to interact with BirdNET models, as well as training and deployment tools for microcontrollers. Make sure to check out our other repositories at `https://github.com/birdnet-team <https://github.com/birdnet-team>`_.


Projects map
------------

We have created an interactive map of projects that use BirdNET. If you are working on a project that uses BirdNET, please let us know and we can add your project to the map.

You can access the map here: `Open projects map <https://birdnet-team.github.io/BirdNET-Analyzer/projects.html>`_

Please refer to the `projects map documentation <usage/projects-map.html>`_ for more information on how to contribute.

License
-------

**Source Code**: The source code for this project is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_

**Models**: The models used in this project are licensed under the `Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0) <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_

Please ensure you review and adhere to the specific license terms provided with each model.

*Please note that all educational and research purposes are considered non-commercial use and it is therefore freely permitted to use BirdNET models in any way.*

Funding
-------

This project is supported by Jake Holshuh (Cornell class of ´69) and The Arthur Vining Davis Foundations.
Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072).
The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E).
In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).