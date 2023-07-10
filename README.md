# HUB_DT

![hub_dt_front](https://github.com/Loken85/HUB_DT/assets/953355/22510738-9396-44fb-90d8-1ccc6363fb0c)

### By Adrian Lindsay
### University of British Columbia


This is the home of HUB-DT: A tool for hierarchical behavioural discovery, utlizing video of experimental lab animals to discover behaviours of interest, and to provide multi-modal analysis by combining behavioural labelling with simultaneously recorded electrophysiology data. 



## How does HUB-DT work?
HUB-DT uses feature tracking from video to produce labelsets of putative behaviour, and to perfrom exploratory analysis on these behavioral labels in conjunction with electrophysiology data. 

![How Does HUB-DT Work?](https://github.com/Loken85/HUB_DT/assets/953355/c0a37568-e7a3-4668-950d-039cfc25d243)

This is accomplished by spectrographic projection of the feature tracking via Morlet wavelets, followed by dimesionality reduction of the resulting space into a low-D density map. This density map is then partitioned via unsupervised hierarchical clustering to provide sets of labelled behaviour. Identified behaviours can be characterised by a 'behavioural prototype', the mean reponse of each of the wavelet frequecies to each tracked feature.

Examples of Behavioural Prototypes:

![full_clust3_waves](https://github.com/Loken85/HUB_DT/assets/953355/7c3aaf10-a811-4fb4-9af2-9c956c3d9b60)![full_clust10_waves](https://github.com/Loken85/HUB_DT/assets/953355/2b1a8657-4234-409a-a54e-f5696be7061d)


HUB-DT provides a full browser based GUI, and is designed to import video tracking data from automated feature tracking sources (like DeepLabCut) or from manual video annotation. It also provides an interface for interactive application of a number of data exploration and machine learning techniques, provides visualisation and plotting tools, and combined analysis of video, behaviour, and electrophysiology data. 

![Example of Labelset plots](https://github.com/Loken85/HUB_DT/assets/953355/915b9d31-23e5-42e9-8c64-44238563cf1c)

![Example of NN Learning using Video](https://github.com/Loken85/HUB_DT/assets/953355/57c4a933-8a05-4b91-b936-06c8b0c07e62)







A full user guide and explanatory paper are currently in progress. The respository is provided to accompany forthcoming publications (in preprint).

