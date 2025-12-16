# Module 4 Data Model: Vision-Language-Action (VLA) Robotics

## Overview

This document defines the data models used in the Vision-Language-Action (VLA) Robotics module. Since this is primarily a documentation module for educational purposes, the data models represent conceptual structures rather than persistent data storage.

## Core Data Structures

### 1. RobotAction

Represents a single action that the robot can perform.

```typescript
interface RobotAction {
  action_type: "navigation" | "manipulation" | "perception" | "interaction";
  parameters: {
    target_location?: string;
    target_object?: string;
    action?: string;
    task?: string;
    [key: string]: any;
  };
  description: string;
  estimated_duration: number; // in seconds
}
```

**Validation Rules:**
- action_type must be one of the defined values
- estimated_duration must be positive
- required parameters depend on action_type

### 2. PlanningResult

Represents the result of LLM-based task planning.

```typescript
interface PlanningResult {
  success: boolean;
  actions: RobotAction[];
  reasoning: string;
  error_message?: string;
}
```

**Validation Rules:**
- If success is true, actions array must not be empty
- If success is false, error_message must be present
- reasoning must be present regardless of success

### 3. VoiceCommand

Represents a structured voice command.

```typescript
interface VoiceCommand {
  original_text: string;
  processed_text: string;
  action: string;
  target_object?: string;
  target_location?: string;
  confidence: number; // 0.0 to 1.0
  timestamp: number; // Unix timestamp
}
```

**Validation Rules:**
- confidence must be between 0.0 and 1.0
- timestamp must be a valid Unix timestamp
- action must not be empty

### 4. ObjectDetection

Represents an object detected in the environment.

```typescript
interface ObjectDetection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  class: string;
  confidence: number; // 0.0 to 1.0
  class_id: number;
}
```

**Validation Rules:**
- bbox coordinates must form a valid rectangle
- confidence must be between 0.0 and 1.0
- class must not be empty

### 5. EnvironmentContext

Represents the current state of the environment.

```typescript
interface EnvironmentContext {
  timestamp: number;
  detections: ObjectDetection[];
  scene_description: string;
  object_locations: {
    [key: string]: ObjectDetection[];
  };
}
```

**Validation Rules:**
- timestamp must be a valid Unix timestamp
- detections array can be empty but must be an array
- scene_description must be a string

### 6. SystemState

Represents the overall state of the VLA system.

```typescript
interface SystemState {
  state: "IDLE" | "LISTENING" | "PLANNING" | "EXECUTING_TASK" | "EMERGENCY_STOP";
  battery_level: number; // 0.0 to 100.0
  current_task: string | null;
  safe_to_operate: boolean;
  timestamp: number;
}
```

**Validation Rules:**
- state must be one of the defined values
- battery_level must be between 0.0 and 100.0
- timestamp must be a valid Unix timestamp

## Message Formats

### ROS 2 Message Format for Action Sequences

```json
{
  "command": "natural language command",
  "actions": [
    {
      "action_type": "navigation",
      "parameters": {"target_location": "kitchen"},
      "description": "Move to kitchen",
      "estimated_duration": 15.0
    }
  ],
  "reasoning": "Explanation of planning decisions"
}
```

### ROS 2 Message Format for Environment Context

```json
{
  "timestamp": 1634567890.123,
  "detections": [
    {
      "bbox": [100, 200, 300, 400],
      "class": "cup",
      "confidence": 0.85,
      "class_id": 41
    }
  ],
  "scene_description": "The scene contains a cup.",
  "object_locations": {
    "cup_center_right": [
      {
        "bbox": [100, 200, 300, 400],
        "class": "cup",
        "confidence": 0.85,
        "class_id": 41
      }
    ]
  }
}
```

## State Transitions

### System State Transitions

```
IDLE ──→ LISTENING ──→ PLANNING ──→ EXECUTING_TASK
  ↑                                    │
  └─────────────────────────────────────┘
  │
EMERGENCY_STOP ←─────────→ (any state)
```

**Transition Rules:**
- IDLE → LISTENING: Voice activation received
- LISTENING → PLANNING: Command received and validated
- PLANNING → EXECUTING_TASK: Action sequence generated
- Any state → EMERGENCY_STOP: Emergency stop activated
- EMERGENCY_STOP → IDLE: Emergency resolved

## Performance Considerations

### Data Size Limits
- Action sequences should be limited to 50 actions to prevent excessive processing
- Detection arrays should be limited to 100 objects to maintain performance
- Message sizes should not exceed 1MB to maintain ROS 2 performance

### Update Frequencies
- Environment context: Updated at 1-5 Hz depending on application
- System state: Updated at 1 Hz minimum
- Action sequences: On-demand based on new commands

## Error Handling

### Data Validation Errors

Common validation errors and their handling:

1. **Invalid action_type**: Log error and skip invalid action
2. **Out-of-range confidence**: Clamp to valid range [0.0, 1.0]
3. **Malformed bounding box**: Discard detection and log warning
4. **Missing required fields**: Use default values or skip entry

### Error Response Format

```typescript
interface ValidationError {
  field: string;
  value: any;
  error: string;
  suggestion?: string;
}
```

## Future Extensions

### Planned Data Model Extensions

1. **Multi-modal Inputs**: Support for additional sensor inputs
2. **Learning History**: Tracking of successful action sequences for learning
3. **User Preferences**: Personalization based on user interaction patterns
4. **Environmental Maps**: 3D spatial data for advanced navigation