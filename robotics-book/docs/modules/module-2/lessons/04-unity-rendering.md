---
title: "Unity High-Fidelity Rendering"
description: "Realistic rendering and human-robot interaction in Unity"
---

# Unity High-Fidelity Rendering

## Overview

Unity provides powerful rendering capabilities for creating high-fidelity visualizations in digital twin environments. This lesson focuses on Unity's rendering pipeline, realistic material creation, lighting systems, and human-robot interaction interfaces that complement Gazebo's physics simulation.

## Learning Objectives

- Configure Unity's rendering pipelines for high-fidelity output
- Create physically-based materials and realistic textures
- Implement advanced lighting systems for realistic environments
- Design intuitive human-robot interaction interfaces
- Optimize rendering performance for real-time applications

## Unity Rendering Pipelines

Unity offers three main rendering pipelines:

1. **Universal Render Pipeline (URP)**: Balanced performance and visual quality
2. **High Definition Render Pipeline (HDRP)**: Maximum visual fidelity for high-end hardware
3. **Built-in Render Pipeline**: Legacy pipeline with basic features

### Selecting the Right Pipeline

For digital twin applications:
- **URP**: Best for real-time simulation with good performance
- **HDRP**: Ideal for high-fidelity visualization and presentation
- **Built-in**: Only for simple visualization or legacy compatibility

## High-Fidelity Material Creation

### Physically-Based Rendering (PBR)

Unity's PBR materials simulate real-world light interaction:

```csharp
// Example: Creating a metallic robot material
Shader shader = Shader.Find("HDRP/Lit");
Material robotMaterial = new Material(shader);

// Metallic workflow parameters
robotMaterial.SetColor("_BaseColor", new Color(0.2f, 0.2f, 0.2f, 1.0f)); // Dark gray base
robotMaterial.SetFloat("_Metallic", 0.9f); // Highly metallic
robotMaterial.SetFloat("_Smoothness", 0.8f); // Smooth surface
robotMaterial.SetTexture("_NormalMap", normalMapTexture); // Surface detail
```

### Material Property Block for Dynamic Changes

```csharp
// Example: Changing robot material based on state
public class RobotMaterialController : MonoBehaviour
{
    public Material baseMaterial;
    private MaterialPropertyBlock propertyBlock;

    void Start()
    {
        propertyBlock = new MaterialPropertyBlock();
    }

    public void SetRobotState(RobotState state)
    {
        Color stateColor = GetStateColor(state);
        propertyBlock.SetColor("_BaseColor", stateColor);
        propertyBlock.SetFloat("_Smoothness", GetStateSmoothness(state));

        GetComponent<Renderer>().SetPropertyBlock(propertyBlock);
    }

    Color GetStateColor(RobotState state)
    {
        switch(state)
        {
            case RobotState.Operational: return Color.green;
            case RobotState.Warning: return Color.yellow;
            case RobotState.Error: return Color.red;
            default: return Color.gray;
        }
    }
}
```

## Advanced Lighting Systems

### Realistic Environment Lighting

```csharp
// Example: Setting up realistic lighting for a digital twin environment
public class EnvironmentLighting : MonoBehaviour
{
    public Light sunLight;
    public ReflectionProbe reflectionProbe;
    public Skybox skyboxMaterial;

    void Start()
    {
        SetupLighting();
        SetupReflections();
    }

    void SetupLighting()
    {
        // Configure directional light to simulate sun
        sunLight.type = LightType.Directional;
        sunLight.color = new Color(0.98f, 0.92f, 0.85f); // Warm sunlight
        sunLight.intensity = 3.14f; // Physically accurate
        sunLight.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        sunLight.shadows = LightShadows.Soft;
        sunLight.shadowStrength = 0.8f;
    }

    void SetupReflections()
    {
        // Configure reflection probe for realistic reflections
        reflectionProbe.mode = ReflectionProbeMode.Realtime;
        reflectionProbe.refreshMode = ReflectionProbeRefreshMode.EveryFrame;
        reflectionProbe.timeSlicingMode = ReflectionProbeTimeSlicingMode.AllFacesAtOnce;
    }
}
```

### Light Probes for Dynamic Objects

```csharp
// Example: Using light probes for moving robots
public class RobotLighting : MonoBehaviour
{
    private LightProbeGroup lightProbeGroup;

    void Start()
    {
        SetupLightProbes();
    }

    void SetupLightProbes()
    {
        // Add light probe group component
        lightProbeGroup = gameObject.AddComponent<LightProbeGroup>();

        // Define probe positions around the robot
        Vector3[] probePositions = {
            transform.position + new Vector3(-0.5f, 0, 0),
            transform.position + new Vector3(0.5f, 0, 0),
            transform.position + new Vector3(0, 0.5f, 0),
            transform.position + new Vector3(0, -0.5f, 0),
            transform.position + new Vector3(0, 0, 0.5f),
            transform.position + new Vector3(0, 0, -0.5f)
        };

        lightProbeGroup.probePositions = probePositions;
    }
}
```

## Human-Robot Interaction (HRI) Interfaces

### Interactive Control Panel

```csharp
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;

public class HRIControlPanel : MonoBehaviour
{
    [Header("Robot Controls")]
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Button moveButton;
    public Button stopButton;
    public Toggle pathPlanningToggle;

    [Header("Status Display")]
    public Text robotNameText;
    public Text batteryLevelText;
    public Text statusText;
    public Image statusIndicator;

    [Header("Events")]
    public UnityEvent<Vector2> onRobotCommand = new UnityEvent<Vector2>();
    public UnityEvent onRobotStop = new UnityEvent();

    void Start()
    {
        SetupUI();
    }

    void SetupUI()
    {
        // Configure sliders with default values
        linearVelocitySlider.minValue = 0f;
        linearVelocitySlider.maxValue = 2.0f;
        linearVelocitySlider.value = 0.5f;

        angularVelocitySlider.minValue = -1.0f;
        angularVelocitySlider.maxValue = 1.0f;
        angularVelocitySlider.value = 0.0f;

        // Register event listeners
        moveButton.onClick.AddListener(OnMoveButtonClicked);
        stopButton.onClick.AddListener(OnStopButtonClicked);
        pathPlanningToggle.onValueChanged.AddListener(OnPathPlanningToggled);
    }

    void OnMoveButtonClicked()
    {
        Vector2 command = new Vector2(linearVelocitySlider.value, angularVelocitySlider.value);
        onRobotCommand.Invoke(command);
        UpdateStatus("Moving", Color.green);
    }

    void OnStopButtonClicked()
    {
        onRobotStop.Invoke();
        UpdateStatus("Stopped", Color.red);
    }

    void OnPathPlanningToggled(bool enabled)
    {
        UpdateStatus(enabled ? "Path Planning: ON" : "Path Planning: OFF",
                    enabled ? Color.yellow : Color.gray);
    }

    void UpdateStatus(string status, Color color)
    {
        statusText.text = status;
        statusIndicator.color = color;
    }
}
```

### VR/AR Interaction Components

```csharp
using UnityEngine;
using UnityEngine.XR;

public class VRInteractionController : MonoBehaviour
{
    [Header("VR Interaction")]
    public GameObject robot;
    public LayerMask interactionLayer;
    public float interactionDistance = 5f;

    private XRNode xrNode;
    private InputDevice inputDevice;

    void Start()
    {
        SetupXRInteraction();
    }

    void SetupXRInteraction()
    {
        xrNode = XRNode.RightHand; // Or LeftHand
        inputDevice = InputDevices.GetDeviceAtXRNode(xrNode);
    }

    void Update()
    {
        UpdateXRInput();
    }

    void UpdateXRInput()
    {
        inputDevice = InputDevices.GetDeviceAtXRNode(xrNode);

        // Check for interaction input
        if (inputDevice.isValid)
        {
            bool triggerPressed = false;
            inputDevice.TryGetFeatureValue(CommonUsages.triggerButton, out triggerPressed);

            if (triggerPressed)
            {
                Ray ray = new Ray(transform.position, transform.forward);
                RaycastHit hit;

                if (Physics.Raycast(ray, out hit, interactionDistance, interactionLayer))
                {
                    HandleInteraction(hit.collider.gameObject);
                }
            }
        }
    }

    void HandleInteraction(GameObject target)
    {
        if (target.CompareTag("Robot"))
        {
            // Open robot control interface
            RobotControlInterface.Instance.ShowControlPanel(target);
        }
        else if (target.CompareTag("Environment"))
        {
            // Handle environment interaction
            EnvironmentController.Instance.InteractWith(target);
        }
    }
}
```

## Integration with ROS 2

### Unity Robotics Package Setup

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityROSGateway : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    private ROSConnection ros;
    private string lidarTopic = "/unity/lidar_scan";
    private string robotCmdTopic = "/unity/robot_cmd";

    void Start()
    {
        ConnectToROS();
        SubscribeToTopics();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIPAddress, rosPort);
    }

    void SubscribeToTopics()
    {
        // Subscribe to robot commands
        ros.Subscribe<TwistMsg>(robotCmdTopic, OnRobotCommandReceived);
    }

    void OnRobotCommandReceived(TwistMsg cmd)
    {
        // Process robot command from ROS
        Vector3 linear = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);

        ProcessRobotCommand(linear, angular);
    }

    void ProcessRobotCommand(Vector3 linear, Vector3 angular)
    {
        // Apply command to Unity robot
        // Implementation depends on your robot's movement system
    }

    public void PublishLidarData(float[] ranges)
    {
        LaserScanMsg lidarMsg = new LaserScanMsg();
        lidarMsg.ranges = ranges;
        lidarMsg.angle_min = -Mathf.PI;
        lidarMsg.angle_max = Mathf.PI;
        lidarMsg.angle_increment = (2 * Mathf.PI) / ranges.Length;
        lidarMsg.time_increment = 0.0f;
        lidarMsg.scan_time = 0.1f;
        lidarMsg.range_min = 0.1f;
        lidarMsg.range_max = 30.0f;

        ros.Publish(lidarTopic, lidarMsg);
    }
}
```

## Performance Optimization

### Level of Detail (LOD) System

```csharp
using UnityEngine;

[RequireComponent(typeof(LODGroup))]
public class RobotLODController : MonoBehaviour
{
    public LOD[] lods;
    public float[] lodDistances = { 10f, 30f, 100f };

    private LODGroup lodGroup;

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        lodGroup = GetComponent<LODGroup>();

        lods = new LOD[lodDistances.Length];
        for (int i = 0; i < lodDistances.Length; i++)
        {
            float screenPercentage = lodDistances[i] / 1000f; // Adjust based on your scale
            lods[i] = new LOD(screenPercentage, GetRenderersForLOD(i));
        }

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    Renderer[] GetRenderersForLOD(int lodLevel)
    {
        // Return appropriate renderers for each LOD level
        // Implementation depends on your robot's structure
        return new Renderer[0];
    }
}
```

### Occlusion Culling Setup

```csharp
// Example: Occlusion area for complex environments
public class EnvironmentOcclusion : MonoBehaviour
{
    public float viewCellSize = 10f;
    public float portalSize = 5f;

    void Start()
    {
        SetupOcclusionAreas();
    }

    void SetupOcclusionAreas()
    {
        // In the Unity editor, you would create OcclusionArea components
        // programmatically or use the built-in occlusion culling tools

        // This is typically done in the editor rather than at runtime
        // for performance reasons
    }
}
```

## Hands-On Lab: Create Interactive HRI Environment

### Step 1: Create Environment

1. Create a new Unity scene for the digital twin environment
2. Import or create a humanoid robot model
3. Set up terrain with realistic textures
4. Configure lighting and reflections

### Step 2: Add HRI Components

1. Create a canvas with robot control interface
2. Implement robot movement based on UI input
3. Add status visualization (battery, sensors, etc.)
4. Create interaction zones for robot control

### Step 3: Test Integration

1. Connect to ROS 2 bridge
2. Verify sensor data visualization
3. Test robot control commands
4. Validate performance metrics

## Troubleshooting Common Issues

### 1. Rendering Performance Issues
- Reduce draw call count using batching
- Optimize shader complexity
- Use appropriate texture compression
- Implement occlusion culling

### 2. HRI Interface Responsiveness
- Optimize UI update frequency
- Use coroutines for non-blocking operations
- Implement proper event handling
- Add visual feedback for user actions

### 3. ROS 2 Connection Problems
- Verify network connectivity
- Check topic names and message types
- Validate message serialization
- Monitor connection status

## Next Steps

In the next lesson, we'll explore the integration between ROS 2, Gazebo, and Unity to create a complete digital twin system where all components work together seamlessly.