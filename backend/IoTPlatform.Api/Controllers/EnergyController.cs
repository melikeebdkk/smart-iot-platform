using Microsoft.AspNetCore.Mvc;
using Npgsql;
using NpgsqlTypes;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;
using MQTTnet;
using MQTTnet.Client;

namespace IoTPlatform.Api.Controllers
{
    [ApiController]
    [Route("api/energy")]
    public class EnergyController : ControllerBase
    {
        private readonly string _connStr = "Host=localhost;Port=5432;Database=iotdb;Username=iotuser;Password=iotpass";
        private static readonly HttpClient _httpClient = new HttpClient();

        // 1️⃣ Son power (Anlık Güç)
        [HttpGet("latest")]
        public IActionResult GetLatest([FromQuery] string deviceId = "ALL")
        {
            using var conn = new NpgsqlConnection(_connStr);
            conn.Open();
            string sql;

            if (deviceId == "ALL")
            {
                sql = @"
                    SELECT 
                        SUM(power) as power,
                        MAX(temperature) as temperature,
                        MAX(humidity) as humidity,
                        MAX(time) as time
                    FROM (
                        SELECT DISTINCT ON (device_id)
                            device_id, power, temperature, humidity, time
                        FROM telemetry
                        ORDER BY device_id, time DESC
                    ) t;
                ";
            }
            else
            {
                sql = @"
                    SELECT power, temperature, humidity, time
                    FROM telemetry
                    WHERE device_id = @deviceId
                    ORDER BY time DESC
                    LIMIT 1;
                ";
            }

            using var cmd = new NpgsqlCommand(sql, conn);
            if (deviceId != "ALL") cmd.Parameters.AddWithValue("deviceId", deviceId);

            using var reader = cmd.ExecuteReader();
            if (!reader.Read()) return NotFound(new { error = "No telemetry data found" });

            return Ok(new
            {
                time = reader.IsDBNull(reader.GetOrdinal("time")) ? (DateTime?)null : reader.GetDateTime(reader.GetOrdinal("time")),
                power = reader.IsDBNull(reader.GetOrdinal("power")) ? 0 : reader.GetDouble(reader.GetOrdinal("power")),
                temperature = reader.IsDBNull(reader.GetOrdinal("temperature")) ? (double?)null : reader.GetDouble(reader.GetOrdinal("temperature")),
                humidity = reader.IsDBNull(reader.GetOrdinal("humidity")) ? (double?)null : reader.GetDouble(reader.GetOrdinal("humidity"))
            });
        }

        // 2️⃣ Bugünün saatlik tüketimi (kWh)
        [HttpGet("daily")]
        public IActionResult GetDaily([FromQuery] string deviceId)
        {
            using var conn = new NpgsqlConnection(_connStr);
            conn.Open();

            string sql = @"
                SELECT 
                  sub.hour as hour,
                  SUM(sub.avg_p) / 1000.0 as total_kwh
                FROM (
                    SELECT 
                      date_trunc('hour', time) as hour,
                      device_id,
                      AVG(power) as avg_p
                    FROM telemetry
                    WHERE (device_id = @d OR @d = 'ALL')
                      AND time >= CURRENT_DATE
                    GROUP BY date_trunc('hour', time), device_id
                ) sub
                GROUP BY sub.hour
                ORDER BY sub.hour";

            var cmd = new NpgsqlCommand(sql, conn);
            cmd.Parameters.AddWithValue("d", deviceId);

            var reader = cmd.ExecuteReader();
            var data = new List<object>();
            while (reader.Read())
            {
                data.Add(new {
                    hour = reader.GetDateTime(0),
                    total = Math.Round(reader.GetDouble(1), 3)
                });
            }
            return Ok(data);
        }

        // 3️⃣ AI Forecast (SQL GÜNCELLENDİ: Sadece en son üretilen tahminleri getirir)
        [HttpGet("forecast")]
        public async Task<IActionResult> GetForecast([FromQuery] string deviceId, [FromQuery] int hours = 6)
        {
            try 
            {
                List<string> targetDevices = new List<string>();
                if (deviceId == "ALL") 
                {
                    targetDevices = await GetDeviceListInternal("home_01");
                }
                else 
                {
                    targetDevices.Add(deviceId);
                }

                if (targetDevices.Count == 0) return Ok(new { forecast = new List<double>(), kwh = 0 });

                await using var conn = new NpgsqlConnection(_connStr);
                await conn.OpenAsync();

                // 🔥 ÖNEMLİ DEĞİŞİKLİK: Sadece en son oluşturulan (MAX created_at) tahmin paketini çekiyoruz
                var sql = @"
                    SELECT 
                        forecast_time,
                        SUM(predicted_power) AS total_power
                    FROM energy_forecasts
                    WHERE created_at = (
                        SELECT MAX(created_at) FROM energy_forecasts
                    )
                    AND device_id = ANY(@devices)
                    GROUP BY forecast_time
                    ORDER BY forecast_time
                    LIMIT @h;
                ";

                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("devices", NpgsqlDbType.Array | NpgsqlDbType.Text, targetDevices.ToArray());
                cmd.Parameters.AddWithValue("h", hours);

                await using var reader = await cmd.ExecuteReaderAsync();
                var list = new List<double>();
                while (await reader.ReadAsync())
                {
                    list.Add(reader.GetDouble(1));
                }

                double[] aggregateForecast = list.ToArray();
                double kwh = 0;

                if (aggregateForecast.Length > 0)
                {
                    kwh = aggregateForecast.Sum() / 1000.0;
                }

                return Ok(new {
                    forecast = aggregateForecast,
                    kwh = Math.Round(kwh, 2),
                    estimatedBill = Math.Round(kwh * 3.5, 2)
                });
            }
            catch (Exception ex) 
            {
                return StatusCode(500, new { error = "Forecast DB Hatası: " + ex.Message });
            }
        }

        // 4️⃣ Cihaz Dağılımı (Pie Chart)
        [HttpGet("by-device")]
        public async Task<IActionResult> GetEnergyByDevice([FromQuery] string parentDevice)
        {
            try {
                await using var conn = new NpgsqlConnection(_connStr);
                await conn.OpenAsync();
                
                var cmd = new NpgsqlCommand(@"
                    SELECT device_id, AVG(power) AS avg_power
                    FROM telemetry
                    WHERE parent_device = @parent 
                      AND time >= (NOW() - INTERVAL '24 hours')
                      AND device_id LIKE 'smart_plug_%'
                    GROUP BY device_id
                    ORDER BY avg_power DESC;", conn);

                cmd.Parameters.AddWithValue("parent", parentDevice ?? "home_01");
                await using var reader = await cmd.ExecuteReaderAsync();
                var result = new List<object>();
                while (await reader.ReadAsync()) {
                    result.Add(new { 
                        device = reader.GetString(0), 
                        power = Math.Round(reader.GetDouble(1), 2) 
                    });
                }
                return Ok(result);
            }
            catch (Exception ex) {
                return StatusCode(500, new { error = ex.Message });
            }
        }

        // 5️⃣ Cihaz Listesi
        [HttpGet("devices")]
        public async Task<IActionResult> GetDevices([FromQuery] string parentDevice)
        {
            var devices = await GetDeviceListInternal(parentDevice ?? "home_01");
            return Ok(devices);
        }

        private async Task<List<string>> GetDeviceListInternal(string parentDevice)
        {
            var devices = new List<string>();
            try {
                await using var conn = new NpgsqlConnection(_connStr);
                await conn.OpenAsync();
                var cmd = new NpgsqlCommand(@"
                    SELECT DISTINCT device_id FROM telemetry
                    WHERE parent_device = @parent AND device_id LIKE 'smart_plug_%'
                    ORDER BY device_id;", conn);

                cmd.Parameters.AddWithValue("parent", parentDevice);
                await using var reader = await cmd.ExecuteReaderAsync();
                while (await reader.ReadAsync()) { 
                    devices.Add(reader.GetString(0)); 
                }
            } catch { }
            return devices;
        }

        [HttpPost("light")]
        public async Task<IActionResult> ControlLight([FromBody] JsonElement body)
        {
            string state = body.GetProperty("state").GetString();
            var factory = new MqttFactory();
            using var client = factory.CreateMqttClient();
            var options = new MqttClientOptionsBuilder()
                .WithTcpServer("localhost", 1883)
                .Build();

            await client.ConnectAsync(options);
            var payload = JsonSerializer.Serialize(new {
                device_id = "light_01",
                parent_device = "home_01",
                device_state = state
            });

            var msg = new MqttApplicationMessageBuilder()
                .WithTopic("iot/commands/light")
                .WithPayload(payload)
                .Build();

            await client.PublishAsync(msg);
            return Ok(new { status = "sent", state });
        }
    }
}