
![Web preview](assets/web_preview.png)

# 3D annotation translation

AI system to transform annotaitons of Computed Tomography (CT) scans made by doctors. Those are critical areas such as spinal chord and esophagus and areas of interest such as GTV and CTV. The annotations are transfered using a affine transform calculated by an algorihm. The annotations (colored in shades of red) are transfered to a given CT without annotations made by doctors.

# Setup

## Install dependencies

    pip install -r requirements.txt

## Run demo

    make website

