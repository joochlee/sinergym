import json

# epJSON 파일 읽기
with open('sinergym/data/buildings/Gwangju_CAV_2410_0924.epJSON', 'r') as f:
    epjson = json.load(f)

# 상세 HVAC 시스템 다이어그램 생성
mermaid_hvac = ["graph LR"]
mermaid_hvac.append("    %% HVAC System Detailed Flow")
mermaid_hvac.append("")

# Zone 정보
zones = list(epjson.get('Zone', {}).keys())
airloops = list(epjson.get('AirLoopHVAC', {}).keys())

# HVAC 구성요소
mermaid_hvac.append("    %% Outdoor Air")
mermaid_hvac.append("    OA[Outdoor Air]")
mermaid_hvac.append("")

# 각 AirLoop에 대해
for i, loop_name in enumerate(airloops[:3], 1):  # 처음 3개만
    prefix = f"L{i}"

    mermaid_hvac.append(f"    %% AirLoop {i}: {loop_name}")
    mermaid_hvac.append(f"    {prefix}_OAMixer[OA Mixer]")
    mermaid_hvac.append(f"    {prefix}_Fan[Fan:OnOff]")
    mermaid_hvac.append(f"    {prefix}_CoolCoil[Cooling Coil DX]")
    mermaid_hvac.append(f"    {prefix}_HeatCoil[Heating Coil]")
    mermaid_hvac.append(f"    {prefix}_Zone[Zone {i}]")
    mermaid_hvac.append(f"    {prefix}_Thermo[Thermostat]")
    mermaid_hvac.append("")

    # 연결
    mermaid_hvac.append(f"    OA --> {prefix}_OAMixer")
    mermaid_hvac.append(f"    {prefix}_OAMixer --> {prefix}_Fan")
    mermaid_hvac.append(f"    {prefix}_Fan --> {prefix}_CoolCoil")
    mermaid_hvac.append(f"    {prefix}_CoolCoil --> {prefix}_HeatCoil")
    mermaid_hvac.append(f"    {prefix}_HeatCoil --> {prefix}_Zone")
    mermaid_hvac.append(f"    {prefix}_Zone -.return.-> {prefix}_OAMixer")
    mermaid_hvac.append(f"    {prefix}_Thermo -.controls.-> {prefix}_HeatCoil")
    mermaid_hvac.append(f"    {prefix}_Thermo -.controls.-> {prefix}_CoolCoil")
    mermaid_hvac.append("")

# 스타일링
mermaid_hvac.append("    %% Styling")
mermaid_hvac.append("    classDef outdoor fill:#E6F3FF,stroke:#333,stroke-width:2px")
mermaid_hvac.append("    classDef equipment fill:#FFF9E6,stroke:#333,stroke-width:2px")
mermaid_hvac.append("    classDef zone fill:#FFE6E6,stroke:#333,stroke-width:2px")
mermaid_hvac.append("    classDef control fill:#F0E6FF,stroke:#333,stroke-width:2px")

# 건물 전체 구조 다이어그램
mermaid_building = ["graph TB"]
mermaid_building.append("    %% Building Structure Overview")
mermaid_building.append("")

# 건물 정보
building_name = list(epjson.get('Building', {}).keys())[0] if 'Building' in epjson else "Building"
num_zones = len(zones)
num_surfaces = len(epjson.get('BuildingSurface:Detailed', {}))
num_windows = len(epjson.get('FenestrationSurface:Detailed', {}))

mermaid_building.append(f"    Building[\"{building_name}<br/>{num_zones}개 Zone\"]")
mermaid_building.append("")

# Zone과 Surface
for i, zone in enumerate(zones[:6], 1):
    zone_id = f"Z{i}"
    mermaid_building.append(f"    {zone_id}[\"Zone: {zone}\"]")
    mermaid_building.append(f"    {zone_id}_Surf[\"Surfaces<br/>Walls, Floors, Roofs\"]")
    mermaid_building.append(f"    {zone_id}_Win[\"Windows\"]")
    mermaid_building.append(f"    {zone_id}_HVAC[\"HVAC Equipment\"]")
    mermaid_building.append(f"    {zone_id}_Load[\"Internal Loads<br/>People, Lights, Equipment\"]")
    mermaid_building.append("")
    mermaid_building.append(f"    Building --> {zone_id}")
    mermaid_building.append(f"    {zone_id} --> {zone_id}_Surf")
    mermaid_building.append(f"    {zone_id} --> {zone_id}_Win")
    mermaid_building.append(f"    {zone_id} --> {zone_id}_HVAC")
    mermaid_building.append(f"    {zone_id} --> {zone_id}_Load")
    mermaid_building.append("")

# 스타일
mermaid_building.append("    %% Styling")
mermaid_building.append("    classDef buildingStyle fill:#FFE6E6,stroke:#333,stroke-width:3px")
mermaid_building.append("    classDef zoneStyle fill:#E6F3FF,stroke:#333,stroke-width:2px")
mermaid_building.append("    classDef componentStyle fill:#E6FFE6,stroke:#333,stroke-width:1px")

# 제어 시스템 다이어그램
mermaid_control = ["graph TB"]
mermaid_control.append("    %% Control System Architecture")
mermaid_control.append("")

# EMS가 있는지 확인
has_ems_sensor = 'EnergyManagementSystem:Sensor' in epjson
has_ems_actuator = 'EnergyManagementSystem:Actuator' in epjson
has_ems_program = 'EnergyManagementSystem:Program' in epjson

if has_ems_sensor or has_ems_actuator or has_ems_program:
    num_sensors = len(epjson.get('EnergyManagementSystem:Sensor', {}))
    num_actuators = len(epjson.get('EnergyManagementSystem:Actuator', {}))
    num_programs = len(epjson.get('EnergyManagementSystem:Program', {}))

    mermaid_control.append(f"    EMS[\"EMS Control System\"]")
    mermaid_control.append(f"    Sensors[\"Sensors<br/>({num_sensors}개)\"]")
    mermaid_control.append(f"    Programs[\"Control Programs<br/>({num_programs}개)\"]")
    mermaid_control.append(f"    Actuators[\"Actuators<br/>({num_actuators}개)\"]")
    mermaid_control.append("")
else:
    mermaid_control.append("    Schedules[\"Schedules<br/>(56개)\"]")
    mermaid_control.append("    Thermostats[\"Thermostats<br/>(5개)\"]")
    mermaid_control.append("    Setpoints[\"Setpoint Managers\"]")
    mermaid_control.append("")

mermaid_control.append("    Zones[\"Building Zones<br/>(6개)\"]")
mermaid_control.append("    HVAC[\"HVAC Systems<br/>(5개)\"]")
mermaid_control.append("    Outputs[\"Outputs & Monitoring\"]")
mermaid_control.append("")

if has_ems_sensor or has_ems_actuator or has_ems_program:
    mermaid_control.append("    Sensors -->|measure| Zones")
    mermaid_control.append("    Sensors -->|measure| HVAC")
    mermaid_control.append("    Sensors --> Programs")
    mermaid_control.append("    Programs --> Actuators")
    mermaid_control.append("    Actuators -->|control| HVAC")
    mermaid_control.append("    EMS --> Sensors")
    mermaid_control.append("    EMS --> Programs")
    mermaid_control.append("    EMS --> Actuators")
else:
    mermaid_control.append("    Schedules -->|define| Thermostats")
    mermaid_control.append("    Schedules -->|define| Setpoints")
    mermaid_control.append("    Thermostats -->|control| HVAC")
    mermaid_control.append("    Setpoints -->|control| HVAC")
    mermaid_control.append("    HVAC -->|condition| Zones")

mermaid_control.append("    Zones --> Outputs")
mermaid_control.append("    HVAC --> Outputs")

# 파일로 저장
output_file = 'MyProject/Docs/Gwangju_CAV_2410_0924_detailed.md'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Gwangju CAV 건물 상세 구조\n\n")

    # 개요
    f.write("## 건물 개요\n\n")
    f.write(f"- **건물명**: {building_name}\n")
    f.write(f"- **Zone 수**: {num_zones}개\n")
    f.write(f"- **Surface 수**: {num_surfaces}개 (벽, 바닥, 천장)\n")
    f.write(f"- **Window 수**: {num_windows}개\n")
    f.write(f"- **HVAC 시스템 수**: {len(airloops)}개 (PSZ-AC)\n")
    f.write(f"- **총 객체 타입**: {len(epjson)}개\n\n")

    # 1. 전체 관계도
    f.write("## 1. 객체 간 전체 관계도\n\n")
    f.write("건물의 주요 구성요소와 그들의 관계를 보여줍니다.\n\n")
    f.write("```mermaid\n")
    with open('MyProject/Docs/Gwangju_CAV_2410_0924_structure.md', 'r') as sf:
        content = sf.read()
        # mermaid 블록만 추출
        start = content.find("```mermaid")
        end = content.find("```", start + 10)
        f.write(content[start+10:end])
    f.write("```\n\n")

    # 2. 건물 구조
    f.write("## 2. 건물 구조\n\n")
    f.write("Zone과 각 Zone의 구성요소를 보여줍니다.\n\n")
    f.write("```mermaid\n")
    f.write("\n".join(mermaid_building))
    f.write("\n```\n\n")

    # 3. HVAC 시스템 흐름
    f.write("## 3. HVAC 시스템 공기 흐름\n\n")
    f.write("외기(Outdoor Air)가 각 Zone으로 공급되는 과정을 보여줍니다.\n\n")
    f.write("```mermaid\n")
    f.write("\n".join(mermaid_hvac))
    f.write("\n```\n\n")

    # 4. 제어 시스템
    f.write("## 4. 제어 시스템\n\n")
    f.write("건물의 제어 구조를 보여줍니다.\n\n")
    f.write("```mermaid\n")
    f.write("\n".join(mermaid_control))
    f.write("\n```\n\n")

    # 주요 객체 목록
    f.write("## 5. 주요 객체 목록\n\n")

    f.write("### HVAC 시스템\n\n")
    f.write(f"- AirLoopHVAC: {len(epjson.get('AirLoopHVAC', {}))}개\n")
    f.write(f"- Fan:OnOff: {len(epjson.get('Fan:OnOff', {}))}개\n")
    f.write(f"- Coil:Cooling:DX:SingleSpeed: {len(epjson.get('Coil:Cooling:DX:SingleSpeed', {}))}개\n")
    f.write(f"- Coil:Heating:Fuel: {len(epjson.get('Coil:Heating:Fuel', {}))}개\n\n")

    f.write("### Zone 구성\n\n")
    f.write(f"- Zone: {len(zones)}개\n")
    for zone in zones:
        f.write(f"  - {zone}\n")
    f.write("\n")

    f.write("### 제어 시스템\n\n")
    f.write(f"- ZoneControl:Thermostat: {len(epjson.get('ZoneControl:Thermostat', {}))}개\n")
    f.write(f"- Schedule:Compact: {len(epjson.get('Schedule:Compact', {}))}개\n")
    f.write(f"- SetpointManager (전체): {len(epjson.get('SetpointManager:MixedAir', {})) + len(epjson.get('SetpointManager:SingleZone:Cooling', {})) + len(epjson.get('SetpointManager:SingleZone:Heating', {}))}개\n\n")

    f.write("### 출력 변수\n\n")
    f.write(f"- Output:Variable: {len(epjson.get('Output:Variable', {}))}개\n")
    f.write(f"- Output:Meter: {len(epjson.get('Output:Meter', {}))}개\n\n")

print(f"✓ 상세 다이어그램 생성 완료: {output_file}")
print("\n생성된 다이어그램:")
print("  1. 객체 간 전체 관계도")
print("  2. 건물 구조 (Zone 및 구성요소)")
print("  3. HVAC 시스템 공기 흐름")
print("  4. 제어 시스템 아키텍처")
