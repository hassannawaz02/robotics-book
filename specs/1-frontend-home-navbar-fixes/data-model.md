# Data Model: Module Cards

## ModuleCard Entity

**Definition**: Represents a module card displayed on the homepage

**Fields**:
- `id`: string - unique identifier for the module
- `title`: string - name of the robotics module
- `icon`: string - path to the SVG icon or image file
- `description`: string - short description of the module
- `link`: string - URL to navigate to when card is clicked (optional)

**Validation rules**:
- Title must not be empty
- Description must be between 20-200 characters
- Icon path must be valid

## Relationships
- Each ModuleCard is independent and displayed as a separate card on the homepage
- ModuleCards are arranged in a grid layout (typically 4 cards in a row on desktop)

## State transitions
- ModuleCard is rendered in a grid on the homepage
- ModuleCard can be clicked to navigate to the relevant module documentation