struct VSOutput {
  float4 pos : SV_Position;
  float2 fragTexCoord;
};

[[vk::binding(0, 0)]]
Sampler2D storageImage;

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
  VSOutput output;
  float x = float(vertexID & 1) * 2.0 - 1.0;
  float y = float(vertexID >> 1) * 2.0 - 1.0;
  output.pos = float4(x, y, 0.0, 1.0);
  output.fragTexCoord = float2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
  return storageImage.Sample(vertIn.fragTexCoord);
}