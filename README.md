# FHLD: Flexible 3D Lanes Detection by Hierarchical Shape Matching

![image](https://user-images.githubusercontent.com/118958290/221391678-70ed0499-40b2-42a3-a4a5-769b88e72a1f.png)

This is an official repository of Flexible 3D Lanes Detection by Hierarchical Shape Matching.

[We are busy reviewing our codes recently, and will release the codes ASAP (no later than summer holiday). Thanks for your patience.]


![image](https://user-images.githubusercontent.com/118958290/209803985-311cde29-b03b-48bc-9674-36a8e0b5af77.png)

Abstract: As one of the basic while vital technologies for HD map construction, 3D lane detection is still an open problem due to varying visual conditions, complex typologies, and strict demands for precision. In this paper, we propose an end-to-end flexible and hierarchical lane detector to predict 3D lane lines from point clouds precisely. Specifically, we design a hierarchical network predicting flexible representations of lane shapes at different levels, simultaneously collecting global instance semantics and avoiding local errors. In the global scope, we propose to regress parametric curves w.r.t adaptive axes that help to make more robust predictions towards complex scenes, while in the local vision we detect the structure of lane segments in each of the dynamic anchor cells sampled along the global predicted curves. Moreover, corresponding global and local shape matching losses and anchor cell generation strategies are designed. Experiments on two datasets show that we overwhelm current top methods under high precision standards, and full ablation studies also verify each part of our method. Our codes and datasets will be released soon.


![image](https://user-images.githubusercontent.com/118958290/209805340-366deda8-e48a-4af1-aeaf-d88042b1aac5.png)

