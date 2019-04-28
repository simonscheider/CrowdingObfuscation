# Simulated Crowding Obfuscation

This repository contains code used in the article: 

*Obfuscating spatial point tracks with simulated crowding*

*Abstract*: 
Spatial point tracks are of concern for an increasing number of analysts studying spatial behaviour patterns and environmental effects. Take an epidemiologist studying the behavior of cyclists and how their health is affected by the city's air quality. The accuracy of such analyses critically depends on the positional accuracy of the tracked points. This poses a serious privacy risk.  Tracks easily reveal a person's identity since the places visited function as fingerprints. Current obfuscation-based privacy protection methods, however, mostly rely on point quality reduction, such as spatial cloaking, grid masking or random noise, and thus render an obfuscated track less useful for exposure assessment. We introduce \textit{simulated crowding} as a point quality preserving obfuscation principle that is based on adding fake points. We suggest two crowding strategies based on \textit{extending} and \textit{masking} a track to defend against inference attacks. We test them across various attack strategies and compare them to state-of-the-art obfuscation techniques both in terms of information loss and attack resilience. Results indicate that simulated crowding provides high resilience against home attacks under constantly low information loss. 

Developers:
* [Simon Scheider](http://geographicknowledge.de)
* [Jiong Wang](https://www.uu.nl/staff/JWang3)
* Maarten Mol

Data and code resources (corresponding to methods used in this article):
-   crowding.py / crowdingRaster.py: Contains code for simulated crowding using vector (slow) and raster (fast) implementations
-   attack.py / attackRaster.py: Contains code for different attack strategies
-   otherMasking.py: Code for Voronoi masking and Random perturbation
-   evaluationExposure.py / evaluationPreserve.py / evaluationReconstruct.py: Evaluation code for information loss and attack efficiency

License: [Creative Commons Attribution Share-Alike 4.0 (CC-BY-SA-4.0)](http://opendefinition.org/licenses/cc-by-sa/)

Last update: 28 April 2019
