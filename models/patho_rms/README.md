# patho_rms @ MHub.ai

For details on the model or how to run the end-to-end pipeline on your data in a single command, visit [mhub.ai/models/patho_rms](https://mhub.ai/models/patho_rms)

This model is trained on histopathy slides with Aveolar or Embryonal Rhabdomyosarcoma tissue present.  The model predicts tissue regions 
of ARMS, ERMS, Stroma/Normal tissue, and necrosis. The input is a slide in DICOM-WSI format.  The output is either a DICoM binary or DICOM fractional segmentation
object, depending on the runtime parameters specified in the config file. 

A further description of this model has been published here: ***


