import json
from collections import defaultdict

# epJSON 파일 읽기
with open('sinergym/data/buildings/Gwangju_CAV_2410_0924.epJSON', 'r') as f:
    epjson = json.load(f)

# 객체 타입과 개수 파악
print("=" * 80)
print("epJSON 파일 구조 분석")
print("=" * 80)
print(f"\n총 객체 타입 수: {len(epjson)}")
print("\n주요 객체 타입과 개수:")
print("-" * 80)

object_types = {}
for obj_type, objects in sorted(epjson.items()):
    count = len(objects) if isinstance(objects, dict) else 1
    object_types[obj_type] = count
    if count <= 20:  # 개수가 적은 것들만 자세히 출력
        print(f"{obj_type}: {count}개")

# 많은 객체들은 따로 그룹화
print("\n대량 객체 타입:")
print("-" * 80)
for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True):
    if count > 20:
        print(f"{obj_type}: {count}개")

# 주요 시스템 식별
print("\n\n" + "=" * 80)
print("주요 시스템 구성요소")
print("=" * 80)

# HVAC 시스템
hvac_types = [k for k in epjson.keys() if 'HVAC' in k or 'Coil' in k or 'Fan' in k]
print(f"\nHVAC 관련 ({len(hvac_types)}개 타입):")
for ht in sorted(hvac_types)[:15]:
    print(f"  - {ht}")

# Zone 관련
zone_types = [k for k in epjson.keys() if 'Zone' in k]
print(f"\nZone 관련 ({len(zone_types)}개 타입):")
for zt in sorted(zone_types)[:15]:
    print(f"  - {zt}")

# Building 관련
building_types = [k for k in epjson.keys() if 'Building' in k or 'Construction' in k or 'Material' in k]
print(f"\n건물 구조 ({len(building_types)}개 타입):")
for bt in sorted(building_types)[:15]:
    print(f"  - {bt}")

# 관계 다이어그램 생성 (Mermaid 형식)
print("\n\n" + "=" * 80)
print("관계 다이어그램 생성 중...")
print("=" * 80)

# 주요 카테고리 정의
categories = {
    'Building': ['Building', 'Site:Location', 'Site:GroundTemperature:BuildingSurface'],
    'Zones': ['Zone'],
    'Surfaces': ['BuildingSurface:Detailed', 'Window', 'Door'],
    'HVAC_System': ['AirLoopHVAC', 'AirLoopHVAC:OutdoorAirSystem'],
    'HVAC_Equipment': ['Fan:ConstantVolume', 'Coil:Heating:Electric', 'Coil:Cooling:DX:SingleSpeed'],
    'Schedules': ['Schedule:Compact', 'ScheduleTypeLimits'],
    'Thermostats': ['ZoneControl:Thermostat', 'ThermostatSetpoint:DualSetpoint'],
    'EMS': ['EnergyManagementSystem:Sensor', 'EnergyManagementSystem:Actuator', 'EnergyManagementSystem:Program'],
    'Output': ['Output:Variable', 'Output:Meter']
}

# Mermaid 다이어그램 생성
mermaid = ["graph TB"]
mermaid.append("    %% EnergyPlus Building Model Structure")
mermaid.append("")

# 노드 정의
node_id_map = {}
node_counter = 0

for category, types in categories.items():
    mermaid.append(f"    %% {category}")
    for obj_type in types:
        if obj_type in epjson:
            count = len(epjson[obj_type])
            node_id = f"node{node_counter}"
            node_counter += 1
            node_id_map[obj_type] = node_id
            # 노드 이름을 짧게 표시
            short_name = obj_type.replace('EnergyManagementSystem:', 'EMS:')
            mermaid.append(f"    {node_id}[\"{short_name}<br/>({count}개)\"]")
    mermaid.append("")

# 주요 관계 표시
mermaid.append("    %% Relationships")
relationships = [
    ('Building', 'Zone', 'contains'),
    ('Zone', 'BuildingSurface:Detailed', 'has surfaces'),
    ('Zone', 'ZoneControl:Thermostat', 'controlled by'),
    ('ZoneControl:Thermostat', 'Schedule:Compact', 'uses schedules'),
    ('AirLoopHVAC', 'Fan:ConstantVolume', 'contains'),
    ('AirLoopHVAC', 'Coil:Heating:Electric', 'contains'),
    ('AirLoopHVAC', 'Coil:Cooling:DX:SingleSpeed', 'contains'),
    ('AirLoopHVAC', 'Zone', 'serves'),
    ('EnergyManagementSystem:Actuator', 'AirLoopHVAC', 'controls'),
    ('EnergyManagementSystem:Sensor', 'Zone', 'monitors'),
    ('Output:Variable', 'Zone', 'monitors'),
    ('Output:Meter', 'AirLoopHVAC', 'monitors')
]

for src_type, dst_type, label in relationships:
    if src_type in node_id_map and dst_type in node_id_map:
        src_id = node_id_map[src_type]
        dst_id = node_id_map[dst_type]
        mermaid.append(f"    {src_id} -->|{label}| {dst_id}")

# 스타일 추가
mermaid.append("")
mermaid.append("    %% Styling")
mermaid.append("    classDef buildingClass fill:#FFE6E6,stroke:#333,stroke-width:2px")
mermaid.append("    classDef zoneClass fill:#E6F3FF,stroke:#333,stroke-width:2px")
mermaid.append("    classDef hvacClass fill:#FFF9E6,stroke:#333,stroke-width:2px")
mermaid.append("    classDef emsClass fill:#FFEBE6,stroke:#333,stroke-width:2px")

# Mermaid 파일 저장
output_path = 'MyProject/Docs/Gwangju_CAV_2410_0924_structure.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# EnergyPlus Building Model Structure\n\n")
    f.write("## Object Relationship Diagram\n\n")
    f.write("```mermaid\n")
    f.write("\n".join(mermaid))
    f.write("\n```\n")

print(f"\n✓ Mermaid 다이어그램 저장됨: {output_path}")

# 상세 연결 관계 분석 (샘플)
print("\n\n" + "=" * 80)
print("상세 객체 연결 예시 (AirLoopHVAC)")
print("=" * 80)

if 'AirLoopHVAC' in epjson:
    for name, obj in list(epjson['AirLoopHVAC'].items())[:3]:
        print(f"\n{name}:")
        for key, value in obj.items():
            if 'name' in key.lower():
                print(f"  → {key}: {value}")

print("\n\n분석 완료!")
