const fs = require('fs');
const path = require('path');

const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf8'));
const { 
  port: PORT,
  detectionThreshold: DETECTION_THRESHOLD,
  personClassId: PERSON_CLASS_ID,
  sceneChangeThreshold: SCENE_CHANGE_THRESHOLD,
  defaultCenter: DEFAULT_CENTER,
  moveNetInputSize
} = packageJson.config;

const MOVE_NET_INPUT_SIZE = [moveNetInputSize, moveNetInputSize];

module.exports = {
  PORT,
  DETECTION_THRESHOLD,
  PERSON_CLASS_ID,
  SCENE_CHANGE_THRESHOLD,
  DEFAULT_CENTER,
  MOVE_NET_INPUT_SIZE,

}; 