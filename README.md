# TorchRoyale
Nerf Miner.

# Installing dependencies

```bash
chmod +x setup.sh
./setup.sh
```

# Overview
This application will provide Clash Royale players with a recommendation with what cards to play and where to place them. The application will be separate from the main Clash Royale client (played on desktop) and will show where to play a card and what card to play in real time.

The application does not interact with the game client, automate gameplay, or be used in live matches. The tool is intended for research and analysis, not for general public use.

# The Setup
The user will have two windows open, one with Clash Royale's game client and the other with Tkinter. The latter will display a gameboard with the game squares, and then overlay an image of the card the user should play.

The user then manually drags the card into the arena, ensuring they are placing it directly onto the tile the model recommended.

**This is in accordance with Clash Royale's fair play policies, which strictly disallow any game automation.**

An extension of this could be to overlay the Tkinter window directly onto the game, but make it completely translucent except for the gridlines.

# Resources
[https://universe.roboflow.com/christoph-feldkircher-pxlqy/clash-royale-card-detection/model/2](https://universe.roboflow.com/christoph-feldkircher-pxlqy/clash-royale-card-detection/model/2)

[https://inference.roboflow.com/quickstart/explore_models/#run-a-private-fine-tuned-model](https://inference.roboflow.com/quickstart/explore_models/#run-a-private-fine-tuned-model)
