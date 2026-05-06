# TorchRoyale

A complete AI Model that analyzes Clash Royale gameplay live videos and plays moves.

## Features

- **Video Analysis**: Processes Clash Royale replay videos using to detect on-field units and hand cards in real-time
- **Decision Transformer**: Transformer that learns strategic patterns from replay data to predict optimal card placements
- **Hybrid Strategy System**: Combines learned transformer with domain-specific heuristic rules for decision-making
- **Automatic Card Tracking**: Stateful hand tracker with evolution and hero detection, mainly handling the Hog 2.6 deck (cannon, fireball, hog rider, ice golem, ice spirit, log, musketeer, skeletons)
- **UI**: An easy-to-use UI with configurable settings, strategy, and live game visualizer tabs
- **App Build**: Portable executable available for Mac users via PyInstaller

## Installation

Download the latest release from [GitHub Releases](https://github.com/Albertlungu/TorchRoyale/releases).

### Requirements

No external installations required! You just need:
- **BlueStacks emulator** (or Android emulator **Note that the application has only been tested with BlueStacks, so other emulators may not work**)
- **An Android device** 

## Known Bugs

- **Strategy Tab**: Matchups with other decks may default to 0.5 for all values and 1.0 for cycle speed. This means that it has not been added yet. 

## Support

For help, please contact [@kashsuks](https://github.com/kashsuks) through [their email](mailto:kashyap@hackclub.help?subject=Issue%20regarding%20TorchRoyale%20on%20GitHub).

To report issues, visit [GitHub Issues page](https://github.com/Albertlungu/TorchRoyale/issues).

## Sources

- **[Clash Royale Official Site](https://clashroyale.fandom.com/wiki/Cards)**: Reference for elixir system, card costs, and deck composition. Constants are defined in `src/constants/cards.py` and `src/constants/game.py`.

- **[PyTorch Documentation](https://pytorch.org/docs/)**: Framework used for implementing the Decision Transformer model, feature encoder, dataset loading, and training loop.

- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**: Used in `src/ocr/detector.py` for detecting game timer and elixir count.

- **[Roboflow Inference Documentation](https://inference.roboflow.com/quickstart/explore_models/#run-a-private-fine-tuned-model)**: Provided guidance on how to run Roboflow models locally. 

- **[PyQt6 Tutorial](https://www.pythonguis.com/pyqt6-tutorial/)**: Documentation used for building the GUI interface. 

- **[Project Structure Inspiration Video](https://www.youtube.com/watch?v=6Gm-pnNieMU)**: Took inspiration from the video on how to outline the project and approach different aspects of the machine learning pipeline architecture.
