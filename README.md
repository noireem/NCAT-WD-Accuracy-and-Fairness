## Project Abstract
Automated Violence Detection Systems (AVDS) are increasingly deployed in sensitive environments (schools, policing), operating on the assumption that object detection algorithms are objective. However, industry-standard models (e.g., YOLOv8) are suspected of disproportionately misclassifying
darker-skinned individuals as armed. This bias stems from "underestimation bias"[1] (unrepresentative training data) and "negative legacy"[1] (subjective labeling of aggressive behavior). While current research focuses on detecting weapons,
this project introduces an intervention: Object Discrimination Training. We investigate if explicitly training models on "confusing objects" (phones, wallets) held by diverse hands reduces False Positives more effectively than standard binary (weapon/no-weapon) training.


## Research Questions
How do historical data labeling practices in standard datasets (UCF-Crime) encode algorithmic bias into the baseline model
Does adding a “Non-Weapon” class filter (trained on diverse hands holding common non weapon objects) increase reliability more than just having weapon classes?

## Tools & Tech Stack
Object Detection Models:
 YOLOv8 (You Only Look Once, Version 8) 
It’s commonly cited in research papers relating to weapon detection (3000+) serving as a solid benchmark for industry grade models.

RT-DETR (Real Time Detection Transformer)
Designed to understand global context (the entire scene) better than most CNNs.

 SSD MobileNet
Quantized model (shrunken models) designed to run on the camera rather than a server.  

Skin Tone Analysis/Clustering: OpenCV + Monk SkinTone Examples (MST-F) Dataset
Currently has the most representative range of skin tones in relation to computational tasks. Is effective in detecting skin tones under different lighting conditions & with bodily obstructions (clothing, objects, masks, etc.)

Cloud & Compute: Azure Machine Learning & Azure Blob Storage
For  training on GPU Clusters & Dataset management

Data Visualization: Plotly (interactive confusion matrices & bias visualization)

Datasets:
Baselines:UCF-Crime  
Intervention: Fair Crime Dataset (Custom dataset) Aggregation of datasets in diverse scenarios
Mock Attack Dataset (indoor setting)
Weapon Detection Rifles vs Umbrellas (synthetic but good quality)
CCTV Pistols
FiDaSS: A Novel Dataset for Firearm Threat Detection in Real-World Scenes
UBI-Fights (Non Weapon Scenarios)
OD-Weapon Detection Weapon & Non Weapon Classifications
https://github.com/tolusophy/Violence_Detection
https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
https://cocodataset.org/#home (Training on common objects)
Methods & Execution

## Part 1: Auditing for hidden impact
Completion date: By March 1st at the latest

This answers the question…
How do historical data labeling practices in standard datasets (UCF-Crime) encode algorithmic bias into the baseline model
Method
Ingest UCF-Crime into YOLOv8
 (if we can find 2-3 other datasets to broaden our scope that’d be ideal)
Filter for High confidence weapon detections (>70%)
Review detected frames & take notes of how the data appears collectively
Apply MST-E classifier to detected persons to tag skin tone buckets by hex code respective to the monk skin tone scalar (1-3,4-7,7-10,undetectable)
Review weapon detections by buckets to identify any patterns that were hidden in the first review

Deliverables: A report on the baseline models’ bias, showing the False Positive Rate per Skin Tone Bucket. Bias Chart that visualizes any disparate differences


## Part 2:  Dataset Intervention 
Completion Date: TBD
This answers the question…
Does adding a “Non-Weapon” class filter (trained on diverse hands holding common non weapon objects) increase reliability more than just having weapon classes?
Method
Aggregating together a Fair Crime Dataset: A dataset representative of different skintones, weapons, non-weapons & environments
Using Multi-label classification  instead of standard labeling:
0: Weapon
1:  Phone
2: Umbrella
3: Wallet
4: Keys
Ensuring that each class has equal representation(MST) of hands holding them
Measure accuracy between this epoch and the results in part one

Deliverables: Append findings onto report in the first deliverable; Data Visualization that contrasts the performance of standard labeling, standard labeling with MST-E implemented, and Multi-Label Classification; Final Paper denoting research.


## Scope & Constraints
Disclaimer: 
The focus of this project is evaluating emerging bias within a simulated environment. With this in mind, there’s a chance the intervention models are too computationally intensive to be added to a CCTV at this moment (also limited to Azure/Colab GPU hours)
This project is also not attempting to optimize surveillance for policing. It’s an audit of existing baseline technologies & highlighting where they could fall short. The end goal is to reduce harm & increase transparency
We will not be identifying individuals (No FRT), only skin phenotypes
Privacy Laws prevent access to raw police CCTV. The datasets used in this project rely on what is publicly available & proxy datasets (movies, Youtube, etc.) which acts as a constraint on real world applications


## Architechture
```mermaid
flowchart TD

subgraph group_data["Data acquisition"]
  node_data_prep["Dataset download<br/>[data_prep.py]"]
  node_local_dataset["Interventiondataset"]
end

subgraph group_baseline["Baseline audit"]
  node_baseline_infer["Baseline inference"]
  node_yolo_infer["YOLO inference<br/>[infer.py]"]
  node_detections["Weapon detections"]
  node_skin_classify["Skin-tone classification"]
end

subgraph group_intervention["Training intervention"]
  node_intervention_data["Intervention dataset<br/>[data.yaml]"]
  node_distill_train["Distillation training<br/>[train.py]"]
  node_trained_model["Trained YOLO model"]
  node_distill_infer["Distilled inference<br/>[infer_and_label.py]"]
  node_intervention_results["Detections and labels"]
end

subgraph group_evaluation["Evaluation"]
  node_fairness["Fairness audit<br/>[fairness_audit.py]"]
  node_visualization["Bias visualization<br/>[viz.py]"]
end

node_researcher(("Researcher"))
node_azure[("Azure Blob")]

node_researcher -->|"starts download"| node_data_prep
node_azure -->|"serves blobs"| node_data_prep
node_data_prep -->|"downloads files"| node_local_dataset
node_local_dataset -->|"provides frames"| node_baseline_infer
node_baseline_infer -->|"saves flagged frames"| node_detections
node_local_dataset -->|"provides images"| node_yolo_infer
node_yolo_infer -->|"writes detections"| node_detections
node_detections -.->|"identifies detections"| node_skin_classify
node_skin_classify -.->|"provides tone buckets"| node_fairness
node_detections -.->|"groups by outcome"| node_fairness
node_fairness -.->|"summarizes bias"| node_visualization
node_intervention_data -->|"configures training"| node_distill_train
node_distill_train -->|"trains student"| node_trained_model
node_local_dataset -->|"provides images"| node_distill_infer
node_trained_model -->|"supplies weights"| node_distill_infer
node_distill_infer -->|"saves detections and labels"| node_intervention_results

click node_data_prep "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/src/data_prep.py"
click node_baseline_infer "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/src/baseline_inference_audit.py"
click node_yolo_infer "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/models/yolov8/infer.py"
click node_skin_classify "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/src/skin_tone_classify.py"
click node_fairness "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/src/fairness_audit.py"
click node_visualization "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/src/viz.py"
click node_intervention_data "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/models/yolov8/data.yaml"
click node_distill_train "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/models/yolov8/train.py"
click node_distill_infer "https://github.com/noireem/ncat-wd-accuracy-and-fairness/blob/main/scripts/infer_and_label.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_data_prep,node_local_dataset toneBlue
class node_baseline_infer,node_yolo_infer,node_detections,node_skin_classify,node_azure toneAmber
class node_intervention_data,node_distill_train,node_trained_model,node_distill_infer,node_intervention_results toneMint
class node_fairness,node_visualization toneRose
class node_researcher toneIndigo
```



## Set-up: Inferencing

### Smoke test
