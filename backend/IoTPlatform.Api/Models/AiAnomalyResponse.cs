using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace IoTPlatform.Api.Models
{
    public class AiAnomalyResponse
    {
        [JsonPropertyName("device_id")]
        public string DeviceId { get; set; } = string.Empty;

        [JsonPropertyName("minutes")]
        public int Minutes { get; set; }

        [JsonPropertyName("anomalies")]
        public List<object> Anomalies { get; set; } = new();
    }
}
