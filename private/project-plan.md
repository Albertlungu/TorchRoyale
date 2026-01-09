**ICS4U** 

**TorchRoyale**  
**Project**

**Plan** 

**Student Name(s): Kashyap Sukshavasi, Johnathan Han, Albert Lungu**  
**Date: January 5th, 2026**

# 

# 1.0 Introduction

This project aims to create a desktop application to give Clash Royale suggestions to the user. The design team, Kashyap Sukshavasi, Johnathan Han, and Albert Lungu has a horizontal management structure meaning there is no specific project manager. The client is Steven Quast and the target audience for the project is mainly for anyone who likes to play Clash Royale as they would want to improve their skills. 

# 1.1 Project Description

This application will provide Clash Royale players with a recommendation with what cards to play and where to place them. The application will be separate from the main Clash Royale client (played on desktop) and will show where to play a card and what card to play in real time. The application does not interact with the game client, automate gameplay, or be used in live matches. The tool is intended for research and analysis, not for general public use. 

# 2.0 Client Requirements 

For this project, The design team has chosen to work with Steven Quast, a physics teacher at Merivale High School, in order for him to utilize the application for his friendly Clash Royale games.

# 2.1 Client Must-haves
The client has identified many must-haves for this application which are outlined below.

Add subject line  
Add must-haves

# 2.2 Client Nice-to-haves
The client has also identified some additional features that would be nice to include in the application which are outlined below. 

Add subject line  
Add nice-to-haves

# 3.0 Project Backlog
The project will be broken down into various subprojects. The project backlog can be seen in Table 3.0.1.

Table 3.0.1: Project Breakdown

| Subproject | Tasks | Difficulty Level |
| :---- | :---- | :---- |
| Setup | Create GitHub repository | 1 |
|  | Set up project documentation | 1 |
| Model Integration | Set up Roboflow API integration | 3 |
|  | Train/fine-tune model on chosen dataset | 5 |
|  | Test model accuracy and detection performance | 3 |
| Desktop Application | Create Tkinter game board UI with grid overlay | 3 |
| | Implement real-time screenshot/screen capture | 5 |
| | Build card recommendation algorithm | 8 |
| | Integrate model with UI for live recommendations | 5 |
| Testing & Debugging | Test end-to-end application flow | 3 |
| | Debug and optimize performance | 5 |
| Final Presentation | Complete presentation slides | 5 |
| | Practice presentation | 1 |

# 4.0 Resources

This section of the project plan will cover the team structure, the online resources used, tools and frameworks, and any other information about the development environment.

## 4.1 Team Structure

The TorchRoyale team consists of three developers: Kashyap Sukshavasi, Johnathan Han, and Albert Lungu. The team operates with a horizontal management structure, meaning there is no designated project manager and all members share equal responsibility for project outcomes.

Work is distributed collaboratively with all team members contributing to all aspects of the project. This approach allows for flexibility in task assignment and ensures that each member gains experience across the full development stack, from model integration to UI development to algorithm design.

Team coordination occurs through Discord for asynchronous communication and in-person collaboration during class time. All members are expected to participate in technical decisions and contribute to code reviews.

## 4.2 Software

The TorchRoyale project will use the following software tools and platforms:

**Programming Language**: Python 3.12

**Desktop Application Framework**: Tkinter (Python standard GUI library)

**Machine Learning & Model**:
- Roboflow API for image inference and card detection
- PyTorch for custom model training and fine-tuning
- Pre-trained image detection model from Roboflow: [https://universe.roboflow.com/christoph-feldkircher-pxlqy/clash-royale-card-detection/model/2](https://universe.roboflow.com/christoph-feldkircher-pxlqy/clash-royale-card-detection/model/2)
- Clash Royale assets dataset: [https://universe.roboflow.com/clashroyale/clash-royale-of3d3](https://universe.roboflow.com/clashroyale/clash-royale-of3d3)

**Development Environment**:
- IDE: VS Code
- Version Control: Git
- Repository Hosting: GitHub
- Preferred terminal lanuage: Bash

**Project Management**:
- GitHub Issues and Projects for sprint backlog and task tracking
- Discord for team communication
- Trello Konbon board for planning specific tasks

**Additional Libraries** (as needed):
- PyTorch 2.9.1 (Most stable version)
- PIL/Pillow for image processing
- NumPy for numerical operations
- Requests for API calls

## 4.3 Hardware

The device for training and running the model(s) will be done on a ASUS TUF Gaming A15 Gaming Laptop. It has 16GB of DDR5 RAM, and 512GB HDD.

# 5.0 Risk Analysis

This section identifies three major risks that could impact the TorchRoyale project's success, along with their preventative measures, corrective actions, and risk levels.

| Risk Matrix | Severity |  |  |  |
| ----- | ----- | :---: | :---: | :---: |
|  **Probability** |  | **Major** | **Moderate** | **Minor** |
|  | **High** | High | High | Medium |
|  | **Medium** | High | Medium | Low |
|  | **Low** | Medium | Low | Low |

## Risk 1: Model Accuracy and Detection Failures

**Description**: The Roboflow detection model may not accurately identify Clash Royale cards or game state in all scenarios, leading to incorrect or missing recommendations.

**Probability**: Medium

**Severity**: Major

**Risk Level**: High

**Preventing it**:
- Test the Roboflow model with many game scenarios before attempting its integration.
- Check detection accuracy across cards, scenarios, and arenas.
- Make minimum accuracy thresholds (e.g., 90% detection rate) before going on to develop the algorithm.

**Fixing it**:
- If there's not enough accuracy, we should fine-tune the model using more training data from the Clash Royale dataset.
- Use confidence thresholds to filter out low-confidence detections.
- Add manual override for users to correct cards that were not correctly identified.
- Consider using multiple models if a single model is inadequate.

## Risk 2: Clash Royale Game Updates Breaking Detection

**Description**: Supercell may update Clash Royale's UI, assets, or game mechanics, which could invalidate the detection model and break the application's ability to correctly identify cards.

**Probability**: Low

**Severity**: Moderate

**Risk Level**: Low

**Preventing it**:
- Monitor Clash Royale's official channels and community forums for updates
- Design the detection layer in a modular way to make model swapping easier
- Keep the specific game version and visual assets the model was trained on

**Fixing it**:
- If game updates break detection, immediately check what changed
- Retrain or fine-tune the model using screenshots from the new game version
- Keep a fallback version of the game that works with the previous game version during retraining
- Ask the Clash Royale community for updated datasets if available

## Risk 3: Repository Changes in Image Detection Model

**Description**: The upstream Roboflow model or detection repository may be updated, modified, or deprecated, potentially breaking compatibility with the TorchRoyale application.

**Probability**: Low

**Severity**: Minor

**Risk Level**: Low

**Preventing it**:
- Fork and version-lock the detection model to maintain a stable copy under team control
- Document the exact model version, API endpoints, and dependencies being used
- Store model weights locally if permitted by Roboflow's licensing terms

**Risk Summary Table**:

| Risk | Probability | Severity | Risk Level |
| :---- | :---- | :---- | :---- |
| Model Accuracy/Detection Failures | Medium | Major | High |
| Game Updates Breaking Detection | Low | Moderate | Medium |
| Repository Changes in Detection Model | Low | Minor | Low |

# 6.0 Team Policies

The team is able to work on the project during and outside the classroom.

Conversations during work time are expected to be on-topic for the most part. We are human and off-topic conversations are expected (and sometimes encouraged).

If a member is unable to attend, they are expected to convey with other members at least 12 hours prior and work can be completed whenever before the end of the sprint.

# 6.1 Time Commitment

All members will work in class and if given the time or if more time is needed, work will be done outside of class. 

# 6.2 Accountability and Communication

The members are expected to be active on Discord to make sure updates reach all members. Members are also responsible for making sure updates reach either other (whether online or in-person) 

# 6.3 Friendship \\(-◡-)/  
The team will keep a light atmosphere while working, listening to K-pop playlists and cracking jokes. 

**Learning Goals**

| P1 | Create a software project plan by producing a software scope document and determining the Assignments, deliverables, and schedule |
| :---- | :---- |
| P5 | Contribute, as a team member, to the planning, development, and production of a large software project |

