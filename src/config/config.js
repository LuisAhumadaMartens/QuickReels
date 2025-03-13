// Direct configuration values instead of reading from package.json
const PORT = 3000;
const DETECTION_THRESHOLD = 0.3;
const PERSON_CLASS_ID = 0;
const SCENE_CHANGE_THRESHOLD = 3000;
const DEFAULT_CENTER = 0.5;
const moveNetInputSize = 192;

const MOVE_NET_INPUT_SIZE = [moveNetInputSize, moveNetInputSize];

module.exports = {
  PORT,
  DETECTION_THRESHOLD,
  PERSON_CLASS_ID,
  SCENE_CHANGE_THRESHOLD,
  DEFAULT_CENTER,
  MOVE_NET_INPUT_SIZE,
}; 