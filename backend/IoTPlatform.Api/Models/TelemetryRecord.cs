using System.Text.Json.Serialization;

namespace IoTPlatform.Api.Models;

public class TelemetryRecord
{
    [JsonPropertyName("device_id")]
    public string Device_Id { get; set; }

    [JsonPropertyName("parent_device")]
    public string Parent_Device { get; set; }   // 🔥 Ev ID (home_01)

    [JsonPropertyName("timestamp")]
    public DateTime Time { get; set; }

    [JsonPropertyName("temperature")]
    public double Temperature { get; set; }

    [JsonPropertyName("humidity")]
    public double Humidity { get; set; }

    [JsonPropertyName("power")]
    public double Power { get; set; }

    [JsonPropertyName("device_state")]
    public string Device_State { get; set; }
}

